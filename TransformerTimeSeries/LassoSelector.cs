using MathNet.Numerics.Statistics;

namespace TransformerTimeSeries;

using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

public class LassoSelector
{
    public static (int[] indicies, double[] betas) SelectLasso(
        Matrix<double> Xraw, Vector<double> y, double lambda, int topK,int maxIter = 1000,double tol = 1e-5)
    {
        int T = Xraw.RowCount;
        int N = Xraw.ColumnCount;
        if (T != y.Count)
            throw new ArgumentException("X and y must have the same number of rows.");
        
        if(topK > N)
            throw new ArgumentException("topK must be less than or equal to the number of features (N).");
        if(topK<0)
            throw new ArgumentException("topK must be greater than or equal to 0.");
        if (lambda < 0)
            throw new ArgumentException("lambda must be non-negative.");
        
        var X = Xraw.Clone();
        

        // Coordinate descent
        var beta = Vector<double>.Build.Dense(N, 0.0);

        for (int iter = 0; iter < maxIter; iter++)
        {
            
            double maxChange = 0;
            for (int j = 0; j < N; j++)
            {
                //Console.WriteLine($"Perc: {iter} / {maxIter} - GradChange: {maxChange}");
                var Xj = X.Column(j);
                // Here compute derivative of the partal error with respect to jth coordinate
                var residual = y - X * beta + beta[j] * Xj;
                double rho = Xj.DotProduct(residual);

                double newBetaJ = SoftThreshold(rho, lambda) / Xj.DotProduct(Xj);
                double change = Math.Abs(newBetaJ - beta[j]);
                maxChange = Math.Max(maxChange, change);
                beta[j] = newBetaJ;
            }
            if (maxChange < tol) break;
        }

        var coeffs = beta.ToArray();
        var topIndices = coeffs
            .Select((val, idx) => new { val, idx })
            .OrderByDescending(x => Math.Abs(x.val))
            .Take(topK)
            .Select(x => x.idx)
            .ToArray();

        return (topIndices, coeffs);
    }

    private static double SoftThreshold(double rho, double lambda)
    {
        if (rho > lambda) return rho - lambda;
        if (rho < -lambda) return rho + lambda;
        return 0.0;
    }
    
    public static Vector<double> MultilinearOLS(Matrix<double> X, Vector<double> y)
    {
        // β = (XᵀX)⁻¹ Xᵀy
        var Xt = X.Transpose();
        var XtX = Xt * X;
        var XtY = Xt * y;
        var beta = XtX.Inverse() * XtY;

        return beta;
    }
}