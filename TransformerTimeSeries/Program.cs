using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace TransformerTimeSeries;

using Deedle;

class Program
{
    public static Frame<DateTime, string> ComputeLogReturns(Frame<DateTime, string> prices)
    {
        // Shift frame down by 1 row
        var shifted = prices.Shift(1);

        // Compute log returns: log(price_t / price_{t-1})
        var returns = prices.Zip<double, double, double>(shifted!, (curr, prev) => { return Math.Log(curr / prev); });

        // Drop the first row with NaNs
        return returns.DropSparseRows();
    }

    
    static void Main(string[] args)
    {
        var topK = 1;
        var classes =
            "AGG,ARKF,ARKG,ARKK,ARKQ,ARKW,BIL,BND,BOTZ,CEW,CORN,CYB,DBC,DIA,EEM,EFA,EUO,EWJ,EWL,EWP,EWS,EWT,EWU,EWW,EWZ,FXB,FXE,FXF,FXY,GLD,HYG,IBB,ICF,IEF,IEI,INDA,IVV,IWM,JJG,LQD,MOAT,MTUM,PALL,PRNT,QQQ,QUAL,ROBO,RWO,RWR,SHY,SLV,SPHD,SPY,TIP,TLT,UNG,USMV,USO,UUP,VBR,VDE,VLUE,VNQ,VOE,VOO,VTI,VTV,VUG,VWO,WEAT,XBI,XLB,XLC,XLE,XLF,XLI,XLK,XLU,XLV";
        var classesList = classes.Split(',').ToList();

        var K = 10;
        var Nclass = classesList.Count / K;
        var assetClasses = Enumerable.Range(0, Nclass).Select(n => classesList.Skip(n * K).Take(K));


        var frame = Deedle.Frame.ReadCsv(
            "/Users/fran/code/TransformerTimeSeries/TransformerTimeSeries/data/factor_timeseries.csv", hasHeaders: true,
            inferTypes: true);
        var df = frame.IndexRows<DateTime>("Date");
        df = df.Where(x => x.Key >= new DateTime(2024, 1, 1));
        var logRet = df.DropSparseRows();

        var target = logRet.GetColumn<double>("XLY");
        var y = Vector<double>.Build.DenseOfArray(target.Values.ToArray<double>());
        y /= y.StandardDeviation();
        var lambdaGrid = LassoSearch.LogSpace(-9, 0, 20);

        var selectedFactors = new List<string>();
        //Select fdactors per class:
        foreach (var group in assetClasses)
        {
            var glist = group.ToList();
            Console.WriteLine($"Group: {string.Join(",", group)}");
            var X = Matrix<double>.Build.DenseOfArray(logRet.Columns[group].ToArray2D<double>());
            var norm = Matrix<double>.Build.DenseOfDiagonalArray(Enumerable.Range(0, X.ColumnCount)
                .Select(i => 1.0 / X.Column(i).StandardDeviation()).ToArray());
            X = X * norm;

            var factors = LassoSearch.FindBestLambda(X, y, lambdaGrid, topK, 0.8);
            Console.WriteLine(
                $"Betas:{string.Join(",", factors.selected.Select(i => factors.coeffs[i]))} {string.Join(",", factors.selected.Select(i => glist[i]))} --> Best Lambda: {factors.bestLambda} ");
            var XSelected = Matrix<double>.Build.DenseOfColumns(factors.selected.Select(i => X.Column(i)));

            var rbetas = LassoSelector.MultilinearOLS(XSelected, y);
            Console.WriteLine(
                $"Betas:{string.Join(",", rbetas)} {string.Join(",", factors.selected.Select(i => glist[i]))} --> Best Lambda: {factors.bestLambda} ");
            var (pred, res) = LassoSearch.PredictOLS(XSelected, y, rbetas);
            
            selectedFactors.AddRange(factors.selected.Select(i => glist[i]));
            // Create index
            var index = logRet.RowIndex;

            // Create Series
            var seriesPred =  new Series<DateTime, double>(index.Keys.Zip(pred).ToDictionary(x => x.Item1, x => x.Item2));
            var seriesRes =  new Series<DateTime, double>(index.Keys.Zip(res).ToDictionary(x => x.Item1, x => x.Item2));
            var seriesactual =  new Series<DateTime, double>(index.Keys.Zip(y).ToDictionary(x => x.Item1, x => x.Item2));
            

            // Build Frame from dictionary
            var frameres = Frame.FromColumns(new Dictionary<string, Series<DateTime, double>> {
                { "Prediction", seriesPred! },
                { "Residuals", seriesRes! },
                { "actual", seriesactual! }
            });
            // Print the Frame
            var filename = string.Join("_", group);
            frameres.SaveCsv($"/Users/fran/tmp/{filename}.csv",includeRowKeys:true);
            var betas = new Series<string,double>(
                glist.Zip(rbetas.ToArray()).ToDictionary(x => x.Item1, x => x.Item2));
            Console.WriteLine(betas);
            Frame.FromRecords(betas.Observations).SaveCsv($"/Users/fran/tmp/{filename}_betas.csv", includeRowKeys: true);
        }
        
        // Save the selected factors to a CSV file
        var Xfinal = Matrix<double>.Build.DenseOfArray(logRet.Columns[selectedFactors].ToArray2D<double>());
        var finalbetas = LassoSelector.MultilinearOLS(Xfinal, y);
        var finalfactors = LassoSearch.PredictOLS(Xfinal, y, finalbetas);
        
        
        // Create Series
        var seriesPred2 =  new Series<DateTime, double>(logRet.RowIndex.Keys.Zip(finalfactors.prediction).ToDictionary(x => x.Item1, x => x.Item2));
        var seriesRes2 =  new Series<DateTime, double>(logRet.RowIndex.Keys.Zip(finalfactors.residuals).ToDictionary(x => x.Item1, x => x.Item2));
        var seriesactual2 =  new Series<DateTime, double>(logRet.RowIndex.Keys.Zip(y).ToDictionary(x => x.Item1, x => x.Item2));
        // Build Frame from dictionary
        var frameres2 = Frame.FromColumns(new Dictionary<string, Series<DateTime, double>> {
            { "Prediction", seriesPred2! },
            { "Residuals", seriesRes2! },
            { "actual", seriesactual2! }
        });
        var filenamef = string.Join("_", selectedFactors);
        frameres2.SaveCsv($"/Users/fran/tmp/{filenamef}_final.csv",includeRowKeys:true);
        var betas2 = new Series<string,double>(
            selectedFactors.Zip(finalbetas).ToDictionary(x => x.Item1, x => x.Item2));
        Frame.FromRecords(betas2.Observations).SaveCsv($"/Users/fran/tmp/{filenamef}_betas.csv",includeRowKeys:true);
        Console.WriteLine("Hello, World!");
    }
}