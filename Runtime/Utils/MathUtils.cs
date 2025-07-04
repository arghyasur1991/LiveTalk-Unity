using System;
using UnityEngine;

namespace LiveTalk.Utils
{
    /// <summary>
    /// Comprehensive mathematical utilities for LiveTalk face animation and geometric processing.
    /// Provides optimized implementations of matrix operations, coordinate transformations, and geometric calculations.
    /// All methods are designed for high-performance real-time face processing and neural network coordinate transformations.
    /// </summary>
    internal static class MathUtils
    {
        #region Geometric Operations

        /// <summary>
        /// Calculates an expanded bounding box around the specified face region with precise integer arithmetic.
        /// </summary>
        /// <param name="boundingBox">The original face bounding box in format (x1, y1, x2, y2)</param>
        /// <param name="expandFactor">The expansion factor to apply (e.g., 1.5 for 50% larger region)</param>
        /// <returns>An expanded rectangular region centered on the original bounding box</returns>
        /// <exception cref="ArgumentException">Thrown when bounding box dimensions are invalid</exception>
        public static Rect ExpandBoundingBox(Vector4 boundingBox, float expandFactor)
        {
            float x = boundingBox.x;
            float y = boundingBox.y;
            float x1 = boundingBox.z;
            float y1 = boundingBox.w;
            
            // Calculate center using integer division
            int xCenter = (int)((x + x1) / 2);
            int yCenter = (int)((y + y1) / 2);
            
            // Calculate original dimensions: w, h = x1-x, y1-y
            float width = x1 - x;
            float height = y1 - y;
            
            // Calculate expansion radius using integer arithmetic for consistency
            int s = (int)(Mathf.Max(width, height) / 2 * expandFactor);
            
            // Return expanded box centered on original: crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
            return new Rect(xCenter - s, yCenter - s, 2 * s, 2 * s);
        }

        /// <summary>
        /// Transforms a set of 2D points using an affine transformation matrix.
        /// </summary>
        /// <param name="pts">The array of 2D points to transform</param>
        /// <param name="M">The 2x3 affine transformation matrix [rotation|translation]</param>
        /// <returns>An array of transformed 2D points in the target coordinate system</returns>
        /// <exception cref="ArgumentNullException">Thrown when pts or M is null</exception>
        /// <exception cref="ArgumentException">Thrown when M is not a 2x3 matrix</exception>
        public static Vector2[] TransformPts(Vector2[] pts, float[,] M)
        {
            if (pts == null)
                throw new ArgumentNullException(nameof(pts));
            if (M == null)
                throw new ArgumentNullException(nameof(M));
            if (M.GetLength(0) != 2 || M.GetLength(1) != 3)
                throw new ArgumentException("Transformation matrix must be 2x3", nameof(M));

            // Apply affine transformation: result = pts @ M[:2, :2].T + M[:2, 2]
            var result = new Vector2[pts.Length];
            
            for (int i = 0; i < pts.Length; i++)
            {
                result[i] = new Vector2(
                    pts[i].x * M[0, 0] + pts[i].y * M[0, 1] + M[0, 2],
                    pts[i].x * M[1, 0] + pts[i].y * M[1, 1] + M[1, 2]
                );
            }
            
            return result;
        }

        /// <summary>
        /// Transforms 2D points using a Unity 4x4 transformation matrix with homogeneous coordinates.
        /// </summary>
        /// <param name="pts">The array of 2D points to transform</param>
        /// <param name="M">The Unity 4x4 transformation matrix</param>
        /// <returns>An array of transformed 2D points</returns>
        /// <exception cref="ArgumentNullException">Thrown when pts is null</exception>
        public static Vector2[] TransPoints2D(Vector2[] pts, Matrix4x4 M)
        {
            if (pts == null)
                throw new ArgumentNullException(nameof(pts));

            var result = new Vector2[pts.Length];
            
            for (int i = 0; i < pts.Length; i++)
            {
                Vector2 pt = pts[i];
                Vector3 newPt = new(pt.x, pt.y, 1.0f);
                Vector3 transformed = M.MultiplyPoint3x4(newPt);
                result[i] = new Vector2(transformed.x, transformed.y);
            }
            
            return result;
        }

        #endregion

        #region Matrix Operations

        /// <summary>
        /// Performs matrix multiplication with exact NumPy compatibility.
        /// </summary>
        /// <param name="a">The left matrix operand</param>
        /// <param name="b">The right matrix operand</param>
        /// <returns>The result of matrix multiplication a × b</returns>
        /// <exception cref="ArgumentNullException">Thrown when either matrix is null</exception>
        /// <exception cref="ArgumentException">Thrown when matrix dimensions are incompatible for multiplication</exception>
        public static float[,] MatrixMultiply(float[,] a, float[,] b)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));

            int rows = a.GetLength(0);
            int cols = b.GetLength(1);
            int inner = a.GetLength(1);
            
            if (inner != b.GetLength(0))
                throw new ArgumentException("Matrix dimensions incompatible for multiplication");
            
            var result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int k = 0; k < inner; k++)
                    {
                        result[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return result;
        }
        
        /// <summary>
        /// Transposes a 2D matrix by swapping rows and columns.
        /// </summary>
        /// <param name="matrix">The matrix to transpose</param>
        /// <returns>The transposed matrix with swapped dimensions</returns>
        /// <exception cref="ArgumentNullException">Thrown when matrix is null</exception>
        public static float[,] TransposeMatrix(float[,] matrix)
        {
            if (matrix == null)
                throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[cols, rows];
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
        }

        /// <summary>
        /// Inverts a 3x3 matrix using the adjugate method.
        /// </summary>
        /// <param name="matrix">The 3x3 matrix to invert</param>
        /// <returns>The inverted matrix</returns>
        /// <exception cref="ArgumentNullException">Thrown when matrix is null</exception>
        /// <exception cref="ArgumentException">Thrown when matrix is not 3x3</exception>
        /// <exception cref="InvalidOperationException">Thrown when matrix is singular (non-invertible)</exception>
        public static float[,] InvertMatrix3x3(float[,] matrix)
        {
            if (matrix == null)
                throw new ArgumentNullException(nameof(matrix));
            if (matrix.GetLength(0) != 3 || matrix.GetLength(1) != 3)
                throw new ArgumentException("Matrix must be 3x3", nameof(matrix));

            float[,] result = new float[3, 3];
            
            // Calculate determinant using cofactor expansion
            float det = matrix[0, 0] * (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1])
                      - matrix[0, 1] * (matrix[1, 0] * matrix[2, 2] - matrix[1, 2] * matrix[2, 0])
                      + matrix[0, 2] * (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]);
            
            if (Mathf.Abs(det) < 1e-6f)
            {
                throw new InvalidOperationException("Matrix is singular and cannot be inverted");
            }
            
            float invDet = 1.0f / det;
            
            // Calculate adjugate matrix elements and multiply by inverse determinant
            result[0, 0] = (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) * invDet;
            result[0, 1] = (matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2]) * invDet;
            result[0, 2] = (matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]) * invDet;
            
            result[1, 0] = (matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2]) * invDet;
            result[1, 1] = (matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]) * invDet;
            result[1, 2] = (matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2]) * invDet;
            
            result[2, 0] = (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]) * invDet;
            result[2, 1] = (matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1]) * invDet;
            result[2, 2] = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) * invDet;
            
            return result;
        }

        /// <summary>
        /// Inverts a 2x3 affine transformation matrix using the analytical method.
        /// </summary>
        /// <param name="M">The 2x3 affine transformation matrix to invert</param>
        /// <returns>The inverted 2x3 affine transformation matrix</returns>
        /// <exception cref="ArgumentNullException">Thrown when M is null</exception>
        /// <exception cref="ArgumentException">Thrown when M is not a 2x3 matrix</exception>
        /// <exception cref="InvalidOperationException">Thrown when the affine matrix is singular</exception>
        public static float[,] InvertAffineTransform(float[,] M)
        {
            if (M == null)
                throw new ArgumentNullException(nameof(M));
            if (M.GetLength(0) != 2 || M.GetLength(1) != 3)
                throw new ArgumentException("Affine matrix must be 2x3", nameof(M));

            // Extract matrix elements: [[a, b, c], [d, e, f]]
            float a = M[0, 0], b = M[0, 1], c = M[0, 2];
            float d = M[1, 0], e = M[1, 1], f = M[1, 2];
            
            // Calculate determinant of the 2x2 linear part
            float det = a * e - b * d;
            
            if (Mathf.Abs(det) < 1e-6f)
            {
                throw new InvalidOperationException("Affine matrix is singular and cannot be inverted");
            }
            
            // Calculate inverse using analytical formula
            // inv = [[e/det, -b/det, (b*f-c*e)/det], [-d/det, a/det, (c*d-a*f)/det]]
            float[,] inv = new float[2, 3];
            inv[0, 0] = e / det;
            inv[0, 1] = -b / det;
            inv[0, 2] = (b * f - c * e) / det;
            inv[1, 0] = -d / det;
            inv[1, 1] = a / det;
            inv[1, 2] = (c * d - a * f) / det;
            
            return inv;
        }

        #endregion

        #region Array Operations

        /// <summary>
        /// Performs element-wise addition of two float arrays.
        /// </summary>
        /// <param name="a">The first array operand</param>
        /// <param name="b">The second array operand</param>
        /// <returns>A new array containing the element-wise sum a[i] + b[i]</returns>
        /// <exception cref="ArgumentNullException">Thrown when either array is null</exception>
        /// <exception cref="ArgumentException">Thrown when arrays have different lengths</exception>
        public static float[] AddArrays(float[] a, float[] b)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));
            if (a.Length != b.Length)
                throw new ArgumentException("Arrays must have the same length");

            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + b[i];
            }
            return result;
        }
        
        /// <summary>
        /// Performs element-wise subtraction of two float arrays.
        /// </summary>
        /// <param name="a">The minuend array</param>
        /// <param name="b">The subtrahend array</param>
        /// <returns>A new array containing the element-wise difference a[i] - b[i]</returns>
        /// <exception cref="ArgumentNullException">Thrown when either array is null</exception>
        /// <exception cref="ArgumentException">Thrown when arrays have different lengths</exception>
        public static float[] SubtractArrays(float[] a, float[] b)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));
            if (a.Length != b.Length)
                throw new ArgumentException("Arrays must have the same length");

            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] - b[i];
            }
            return result;
        }
        
        /// <summary>
        /// Performs element-wise multiplication of two float arrays.
        /// </summary>
        /// <param name="a">The first array operand</param>
        /// <param name="b">The second array operand</param>
        /// <returns>A new array containing the element-wise product a[i] * b[i]</returns>
        /// <exception cref="ArgumentNullException">Thrown when either array is null</exception>
        /// <exception cref="ArgumentException">Thrown when arrays have different lengths</exception>
        public static float[] MultiplyArrays(float[] a, float[] b)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));
            if (a.Length != b.Length)
                throw new ArgumentException("Arrays must have the same length");

            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return result;
        }
        
        /// <summary>
        /// Performs element-wise division of two float arrays.
        /// </summary>
        /// <param name="a">The dividend array</param>
        /// <param name="b">The divisor array</param>
        /// <returns>A new array containing the element-wise quotient a[i] / b[i]</returns>
        /// <exception cref="ArgumentNullException">Thrown when either array is null</exception>
        /// <exception cref="ArgumentException">Thrown when arrays have different lengths</exception>
        /// <exception cref="DivideByZeroException">Thrown when any element in b is zero</exception>
        public static float[] DivideArrays(float[] a, float[] b)
        {
            if (a == null)
                throw new ArgumentNullException(nameof(a));
            if (b == null)
                throw new ArgumentNullException(nameof(b));
            if (a.Length != b.Length)
                throw new ArgumentException("Arrays must have the same length");

            var result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                if (Mathf.Abs(b[i]) < 1e-6f)
                    throw new DivideByZeroException($"Division by zero at index {i}");
                result[i] = a[i] / b[i];
            }
            return result;
        }

        #endregion

        #region Rotation and Transform Operations

        /// <summary>
        /// Constructs a 3D rotation matrix from Euler angles using the ZYX rotation sequence.
        /// </summary>
        /// <param name="pitch">The pitch rotation angles in degrees (X-axis rotation)</param>
        /// <param name="yaw">The yaw rotation angles in degrees (Y-axis rotation)</param>
        /// <param name="roll">The roll rotation angles in degrees (Z-axis rotation)</param>
        /// <returns>A 3x3 rotation matrix representing the combined rotations</returns>
        /// <exception cref="ArgumentNullException">Thrown when any angle array is null</exception>
        /// <exception cref="ArgumentException">Thrown when angle arrays are empty or have different lengths</exception>
        public static float[,] GetRotationMatrix(float[] pitch, float[] yaw, float[] roll)
        {
            if (pitch == null)
                throw new ArgumentNullException(nameof(pitch));
            if (yaw == null)
                throw new ArgumentNullException(nameof(yaw));
            if (roll == null)
                throw new ArgumentNullException(nameof(roll));
            if (pitch.Length == 0 || yaw.Length == 0 || roll.Length == 0)
                throw new ArgumentException("Angle arrays cannot be empty");

            // Convert degrees to radians for the first element (batch processing)
            float p = pitch[0] * Mathf.Deg2Rad;
            float y = yaw[0] * Mathf.Deg2Rad;
            float r = roll[0] * Mathf.Deg2Rad;
            
            // Precompute trigonometric values for efficiency
            float cos_p = Mathf.Cos(p);
            float sin_p = Mathf.Sin(p);
            float cos_y = Mathf.Cos(y);
            float sin_y = Mathf.Sin(y);
            float cos_r = Mathf.Cos(r);
            float sin_r = Mathf.Sin(r);
            
            // Construct individual rotation matrices following right-hand rule
            // X-axis rotation (pitch)
            var rotX = new float[3, 3] {
                { 1, 0, 0 },
                { 0, cos_p, -sin_p },
                { 0, sin_p, cos_p }
            };
            
            // Y-axis rotation (yaw)
            var rotY = new float[3, 3] {
                { cos_y, 0, sin_y },
                { 0, 1, 0 },
                { -sin_y, 0, cos_y }
            };

            // Z-axis rotation (roll)
            var rotZ = new float[3, 3] {
                { cos_r, -sin_r, 0 },
                { sin_r, cos_r, 0 },
                { 0, 0, 1 }
            };
            
            // Combine rotations in ZYX order: rot = rot_z @ rot_y @ rot_x
            var rotZY = MatrixMultiply(rotZ, rotY);
            var rot = MatrixMultiply(rotZY, rotX);
            return TransposeMatrix(rot);
        }

        #endregion

        #region Coordinate System Conversions

        /// <summary>
        /// Converts a 2x3 affine transformation matrix to Unity's 4x4 Matrix4x4 format.
        /// </summary>
        /// <param name="MInv">The 2x3 inverse affine transformation matrix to convert</param>
        /// <returns>A Unity Matrix4x4 representing the same transformation in homogeneous coordinates</returns>
        /// <exception cref="ArgumentNullException">Thrown when MInv is null</exception>
        /// <exception cref="ArgumentException">Thrown when MInv is not a 2x3 matrix</exception>
        public static Matrix4x4 GetCropTransform(float[,] MInv)
        {
            if (MInv == null)
                throw new ArgumentNullException(nameof(MInv));
            if (MInv.GetLength(0) != 2 || MInv.GetLength(1) != 3)
                throw new ArgumentException("Matrix must be 2x3 affine transformation", nameof(MInv));

            // Construct Unity Matrix4x4 from 2x3 affine matrix
            // Map 2D transformation to 3D homogeneous coordinates
            var Mo2c = Matrix4x4.identity;
            Mo2c.m00 = MInv[0, 0]; Mo2c.m01 = MInv[0, 1]; Mo2c.m03 = MInv[0, 2];
            Mo2c.m10 = MInv[1, 0]; Mo2c.m11 = MInv[1, 1]; Mo2c.m13 = MInv[1, 2];
            Mo2c.m20 = 0f; Mo2c.m21 = 0f; Mo2c.m22 = 1f; Mo2c.m23 = 0f;
            Mo2c.m30 = 0f; Mo2c.m31 = 0f; Mo2c.m32 = 0f; Mo2c.m33 = 1f;
            return Mo2c;
        }

        #endregion
    }
}
