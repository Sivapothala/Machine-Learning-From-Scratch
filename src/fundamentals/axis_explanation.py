import numpy as np
import pandas as pd

def explain_axis():
    """
    Explanation of axis parameter in NumPy and Pandas with examples.
    """
    # Create a sample 2D array
    arr = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    # Create a sample DataFrame
    df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
    
    print("Original Array:")
    print(arr)
    print("\nOriginal DataFrame:")
    print(df)
    
    # NumPy axis explanation
    print("\n=== NumPy Axis Explanation ===")
    print("\naxis=0 (Down the rows):")
    print("Sum along axis=0:", np.sum(arr, axis=0))
    print("This means: sum each column (1+4+7, 2+5+8, 3+6+9)")
    
    print("\naxis=1 (Across the columns):")
    print("Sum along axis=1:", np.sum(arr, axis=1))
    print("This means: sum each row (1+2+3, 4+5+6, 7+8+9)")
    
    # Pandas axis explanation
    print("\n=== Pandas Axis Explanation ===")
    print("\naxis=0 (Index/Row axis):")
    print("Sum along axis=0:")
    print(df.sum(axis=0))
    print("This means: sum each column (same as NumPy axis=0)")
    
    print("\naxis=1 (Column axis):")
    print("Sum along axis=1:")
    print(df.sum(axis=1))
    print("This means: sum each row (same as NumPy axis=1)")
    
    # Visual representation
    print("\n=== Visual Representation ===")
    print("For a 2D array/DataFrame:")
    print("axis=0: ↓ (down the rows)")
    print("axis=1: → (across the columns)")
    
    # Common operations and their axis
    print("\n=== Common Operations and Their Axis ===")
    print("1. Mean of each column: axis=0")
    print("2. Mean of each row: axis=1")
    print("3. Concatenate vertically: axis=0")
    print("4. Concatenate horizontally: axis=1")
    
    # Key points to remember
    print("\n=== Key Points to Remember ===")
    print("1. In NumPy and Pandas, axis=0 always refers to the row axis")
    print("2. axis=1 always refers to the column axis")
    print("3. The operation is performed along the specified axis")
    print("4. The result's shape will have the specified axis removed")
    
    # Example with different shapes
    print("\n=== Example with Different Shapes ===")
    arr_3d = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    print("3D array shape:", arr_3d.shape)
    print("Sum along axis=0:", np.sum(arr_3d, axis=0).shape)
    print("Sum along axis=1:", np.sum(arr_3d, axis=1).shape)
    print("Sum along axis=2:", np.sum(arr_3d, axis=2).shape)

if __name__ == "__main__":
    explain_axis() 