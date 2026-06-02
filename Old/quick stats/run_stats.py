import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load the data
df = pd.read_csv('face_data.csv')

# 2. Clean the data 
target_columns = ['ZeroFace', 'OneFace', 'TwoFace', 'ThreeFace', 'FourFace']
essential_columns = ['GroupSize', 'Spatiality', 'Environment', 'TotalFrames'] + target_columns
df = df.dropna(subset=essential_columns)

df['GroupSize'] = df['GroupSize'].astype(str)
df['LogFrames'] = np.log(df['TotalFrames'])

# Create an empty list to store the results
all_results = []

# 3. Loop through each target and run the analysis
print("Starting analysis...")

for target in target_columns:
    print(f"Running model for: {target}...")
    formula = f"{target} ~ C(GroupSize) + C(Spatiality) + C(Environment)"
    
    try:
        model = smf.glm(formula=formula, 
                        data=df, 
                        family=sm.families.NegativeBinomial(), 
                        offset=df['LogFrames']).fit()
        
        # Extract the pure math for the AI to read
        res_df = pd.DataFrame({
            'Target_Face_Count': target,
            'Condition': model.params.index,
            'Coefficient_LogOdds': model.params.values,
            'P_Value': model.pvalues.values,
            'Std_Error': model.bse.values
        })
        
        all_results.append(res_df)
        print("  -> Success!")
        
    except Exception as e:
        print(f"  -> WARNING: Failed to converge. Error: {e}")

# 4. Combine all the results and save to a CSV
if all_results:
    final_output = pd.concat(all_results, ignore_index=True)
    
    # Round the numbers to make the file smaller and cleaner
    final_output['Coefficient_LogOdds'] = final_output['Coefficient_LogOdds'].round(4)
    final_output['P_Value'] = final_output['P_Value'].round(5)
    final_output['Std_Error'] = final_output['Std_Error'].round(4)
    
    # Save the file
    final_output.to_csv('regression_results_for_ai.csv', index=False)
    print("\nAll done! Results saved to 'regression_results_for_ai.csv'.")
else:
    print("\nNo models completed successfully. No file saved.")