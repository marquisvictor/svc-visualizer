def test_merge_index():
    df1 = pd.DataFrame({'A': [1, 2]}, index=['x', 'y'])
    df2 = pd.DataFrame({'B': [3, 4]}, index=['x', 'y'])
    merged = merge_index(df1, df2)
    assert (merged.columns == ['A', 'B']).all()  # Checking if the merged DataFrame has both 'A' and 'B' columns

def test_mask_insignificant_t_values():
    df = pd.DataFrame({
        'beta_A': [0.1, 0.2],
        'std_A': [0.05, 0.1]
    }, index=['x', 'y'])

    masked = mask_insignificant_t_values(df)

    # Add appropriate assertions based on the expected outcome, such as:
    assert 't_beta_A' in masked.columns  # Check if the t-value column has been added
    # Add more assertions to check the values within the t-value column, ensuring insignificant t-values are masked
