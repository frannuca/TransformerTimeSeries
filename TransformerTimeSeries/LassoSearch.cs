using MathNet.Numerics.LinearAlgebra;

namespace TransformerTimeSeries;

public static class LassoSearch
{
    public static double[] LogSpace(double startExp, double endExp, int num)
    {
        double[] result = new double[num];
        double step = (endExp - startExp) / (num - 1);

        for (int i = 0; i < num; i++)
        {
            result[i] = Math.Pow(10, startExp + step * i);
        }

        return result;
    }
    public static (double bestLambda, int[] selected, double[] coeffs) FindBestLambda(
        Matrix<double> X, Vector<double> y, double[] lambdaGrid, int topK, double trainRatio = 0.8)
    {
        int T = X.RowCount;
        int split = (int)(T * trainRatio);
        int N = X.ColumnCount;

        // Split into training and validation
        Matrix<double> Xtrain = X.SubMatrix(0, split, 0, N);
        Vector<double> ytrain = Vector<double>.Build.DenseOfArray(y.Take(split).ToArray());
        Matrix<double> Xval = X.SubMatrix(split, T-split, 0,N);
        Vector<double> yval = Vector<double>.Build.DenseOfArray(y.Skip(split).ToArray());

        double bestMSE = double.PositiveInfinity;
        double bestLambda = 0;
        double[] bestCoeffs = null;
        int[] bestSelected = null;

        foreach (var lambda in lambdaGrid)
        {
            //Console.WriteLine("Lambda: " + lambda);
            var (selected, coeffs) = LassoSelector.SelectLasso(Xtrain, ytrain, lambda, topK);

            var betas = Vector<double>.Build.DenseOfArray( selected.Select( i =>coeffs[i]).ToArray());
            Matrix<double> SelectedFactorsMatrix = SubsetMatrix(Xval, selected); 
          
            var (pred,res) = PredictOLS(SelectedFactorsMatrix, yval, betas);
            
            var mse = res.Select(x => x*x).Average();

            if (mse < bestMSE)
            {
                bestMSE = mse;
                bestLambda = lambda;
                bestCoeffs = coeffs;
                bestSelected = selected;
            }
        }

        return (bestLambda, bestSelected, bestCoeffs);
    }

    public static double[,] SliceRows(double[,] X, int fromRow, int toRow)
    {
        int cols = X.GetLength(1);
        var result = new double[toRow - fromRow, cols];
        for (int i = fromRow; i < toRow; i++)
        for (int j = 0; j < cols; j++)
            result[i - fromRow, j] = X[i, j];
        return result;
    }

    public static Matrix<double> SubsetMatrix(Matrix<double>  X, int[] columns)
    {
        return Matrix<double>.Build.DenseOfColumnVectors(columns.Select(i => X.Column(i)));
    }

    public static (Vector<double> prediction, Vector<double> residuals)  PredictOLS(Matrix<double> X, Vector<double> y, Vector<double> betas)
    {
        var pred = X.Multiply(betas.ToColumnMatrix());
        var residuals = y - pred.Column(0);
        if(pred.ColumnCount > 1)
            throw new ArgumentException("Prediction matrix has more than one column.");

        return (prediction: pred.Column(0), residuals: residuals);
    }
}