import pandas as pd 
import numpy as np 

# 1. Loading data
# Specify the [path] (the location where the file is placed)
excel_file ="simulations/demographics.xlsx"

# It reads by specifying a particular [sheet] of the Excel file.
df_age =pd .read_excel (excel_file ,sheet_name ='Age data')
df_income =pd .read_excel (excel_file ,sheet_name ='Income data by age',index_col =0 )


# Rows with the total (the bottom row) are excluded because they are not used in the calculation.
df_age =df_age [df_age ['Age']!='Total number'].copy ()
df_age ['Age']=df_age ['Age'].astype (int )

# 2. Probabilistically generate the ages of 1,000 people
# Using [random numbers] (arbitrary numbers), we will create 1000 ages based on proportions.
ages_list =df_age ['Age'].values 
probabilities =df_age ['Ratio'].values 
# We will fine-tune it so that the sum of the probabilities becomes 1.
probabilities =probabilities /probabilities .sum ()

generated_ages =np .random .choice (ages_list ,size =1000 ,p =probabilities )

# 3. Determine the annual income corresponding to each age (this part is the same as last time, but progress will be displayed)
print ("Generating data...")
persona_data =[]
age_to_group =dict (zip (df_age ['Age'],df_age ['Income group']))
income_bands =df_income .columns .tolist ()

for i ,age in enumerate (generated_ages ):
    group =age_to_group [age ]
    income_probs =df_income .loc [group ].values 
    income_probs =income_probs /income_probs .sum ()
    selected_income =np .random .choice (income_bands ,p =income_probs )

    persona_data .append ({
    'id':i +1 ,
    'Age':age ,
    'Annual income':selected_income 
    })

    # 4. Save as CSV file
    # I changed it to be saved inside the simulations folder.
df_result =pd .DataFrame (persona_data )
output_path ="simulations/customer_persona_sheet.csv"
df_result .to_csv (output_path ,index =False ,encoding ='utf-8-sig')

print ("-"*30 )
print (f"[Completed] The file has been created!")
print (f"Location: {output_path }")
print ("-"*30 )