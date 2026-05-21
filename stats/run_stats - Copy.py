import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 1. Load and clean the data
df = pd.read_csv('face_data.csv')

target_columns = ['ZeroFace', 'OneFace', 'TwoFace', 'ThreeFace', 'FourFace']
essential_columns = ['GroupSize', 'Spatiality', 'Environment', 'TotalFrames'] + target_columns
df = df.dropna(subset=essential_columns)

df['GroupSize'] = df['GroupSize'].astype(str)
df['GroupSize_int'] = df['GroupSize'].astype(int)
df['LogFrames'] = np.log(df['TotalFrames'])

# 2. Re-bin the data into Ratio Categories
print("Calculating Group Ratios...")
df['Ratio_Zero'] = df['ZeroFace']
df['Ratio_Partial'] = 0
df['Ratio_Full'] = 0
df['Ratio_SelfView'] = 0

# Logic for pairs (N=2)
mask2 = df['GroupSize_int'] == 2
df.loc[mask2, 'Ratio_Full'] = df.loc[mask2, 'OneFace']
df.loc[mask2, 'Ratio_SelfView'] = df.loc[mask2, 'TwoFace'] + df.loc[mask2, 'ThreeFace'] + df.loc[mask2, 'FourFace']

# Logic for triads (N=3)
mask3 = df['GroupSize_int'] == 3
df.loc[mask3, 'Ratio_Partial'] = df.loc[mask3, 'OneFace']
df.loc[mask3, 'Ratio_Full'] = df.loc[mask3, 'TwoFace']
df.loc[mask3, 'Ratio_SelfView'] = df.loc[mask3, 'ThreeFace'] + df.loc[mask3, 'FourFace']

# Logic for quads (N=4)
mask4 = df['GroupSize_int'] == 4
df.loc[mask4, 'Ratio_Partial'] = df.loc[mask4, 'OneFace'] + df.loc[mask4, 'TwoFace']
df.loc[mask4, 'Ratio_Full'] = df.loc[mask4, 'ThreeFace']
df.loc[mask4, 'Ratio_SelfView'] = df.loc[mask4, 'FourFace']

# 3. Run the analyses on the new semantic targets
ratio_targets = ['Ratio_Zero', 'Ratio_Partial', 'Ratio_Full', 'Ratio_SelfView']
all_results = []

print("Starting analysis on Ratio groupings...")

for target in ratio_targets:
    print(f"Running model for: {target}...")
    
    # Note: 'Partial' is impossible for a 2-person group (you either see them or you don't).
    # To prevent math errors, we exclude GroupSize 2 when analyzing the Partial category.
    if target == 'Ratio_Partial':
        model_df = df[df['GroupSize_int'] > 2].copy()
    else:
        model_df = df.copy()

    formula = f"{target} ~ C(GroupSize) + C(Spatiality) + C(Environment)"
    
    try:
        model = smf.glm(formula=formula, 
                        data=model_df, 
                        family=sm.families.NegativeBinomial(), 
                        offset=model_df['LogFrames']).fit()
        
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

# 4. Save the results
if all_results:
    final_output = pd.concat(all_results, ignore_index=True)
    final_output['Coefficient_LogOdds'] = final_output['Coefficient_LogOdds'].round(4)
    final_output['P_Value'] = final_output['P_Value'].round(5)
    final_output['Std_Error'] = final_output['Std_Error'].round(4)
    final_output.to_csv('ratio_results_for_ai.csv', index=False)
    print("\nAll done! Results saved to 'ratio_results_for_ai.csv'.")
else:
    print("\nNo models completed successfully. No file saved.")