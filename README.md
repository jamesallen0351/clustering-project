# clustering_project_README.md

# Zillow: What is driving the errors in the Zestimates?

### James Allen

## Executive Summary

- This is where I will write my executive summary

### Counties and Tax %:
- Los Angeles: 1.38 %
- Orange: 1.21 %
- Ventura: 1.19%

## Project Overview

- For this project you will continue working with the zillow dataset. Continue to use the 2017 properties and predictions data for single unit / single family homes.

- In addition to continuing work on your previous project, you should incorporate clustering methodologies on this project.

- Your audience for this project is a data science team. The presentation will consist of a notebook demo of the discoveries you made and work you have done related to uncovering what the drivers of the error in the zestimate is.

## Business Goals

- Predict the value of single unit properties duirng May - August, 2017
- Identify states and counties of the properties and the distribution of tax rates
- Presentation about findings to Zillow Data Science Team 

## Deliverables

1. A github repository with the following contents:

    - A clearly named final notebook. This notebook will be what you present and should contain plenty of markdown documentation and cleaned up code.
   
    - A README that explains what the project is, how to reproduce you work, and your notes from project planning.
    
    - A Python module or modules that automate the data acquisistion and preparation process. These modules should be imported and used in your final notebook.

2.  Further project requirements:

    - Data Acquisition: Data is collected from the codeup cloud database with an appropriate SQL query
    
    - Data Prep: 
        - Column data types are appropriate for the data they contain
        - Missing values are investigated and handled
        - Outliers are investigated and handled
    
    - Exploration:
        - The interaction between independent variables and the target variable is explored using visualization and statistical testing
        - Clustering is used to explore the data. A conclusion, supported by statistical testing and visualization, is drawn on whether or not the clusters are helpful/useful. At least 3 combinations of features for clustering should be tried.
    - Modeling:
        - At least 4 different models are created and their performance is compared. One model is the distinct combination of algorithm, hyperparameters, and features.
        - Best practices on data splitting are followed
        
    - The final notebook has a good title and the documentation within is sufficiently explanatory and of high quality
    - Decisions and judment calls are made and explained/documented
    - All python code is of high quality

## Data Dictionary

| Column Name                  | Renamed   | Info                                            |
|------------------------------|-----------|-------------------------------------------------|
| parcelid                     | N/A       | ID of the property (unique)                     |
| bathroomcnt                  | baths     | number of bathrooms                             |
| bedroomcnt                   | beds      | number of bedrooms                              |
| calculatedfinishedsquarefeet | sqft      | number of square feet                           |
| fips                         | N/A       | FIPS code (for county)                          |
| propertylandusetypeid        | N/A       | Type of property                                |
| yearbuilt                    | N/A       | The year the property was built                 |
| taxvaluedollarcnt            | tax_value | Property's tax value in dollars                 |
| taxamount                    | tax_amount| amount of tax on property                       |
| tax_rate                     | N/A       | tax_rate on property                            |


## Data Science Pipeline

- Project Planning
    - Goal: leave this section with (at least the outline of) a plan for the project documented in your README.md file.

- Acquire
    - Goal: leave this section with a dataframe ready to prepare.
    - Create an acquire.py file the reproducible component for gathering data from a database using SQL and reading it into a pandas DataFrame.

- Prepare
    - Goal: leave this section with a dataset that is split into train, validate, and test ready to be analyzed. Make sure data types are appropriate and missing values have been addressed, as have any data integrity issues.
    - Create a prep.pyfile as the reproducible component that handles missing values, fixes data integrity issues, changes data types, scales data, etc.

- Data Exploration
    - Goal: The findings from your analysis should provide you with answers to the specific questions your customer asked that will be used in your final report as well as information to move forward toward building a model.
        - Run at least 1 statistical test

        - Make sure to summarize your takeaways and conclusions. 


- Modeling
    - Goal: develop a regression model that performs better than a baseline.
    - feature engineering
        - Which features should be included in your model?



## Key Findings and Takeaways

- This is where I will write my key findings and takeaways

- Counties and Tax include:
    - Los Angeles : 1.38 %
    - Orange : 1.21 %
    - Ventura : 1.19 %

## Next steps

- With more time I would like to:

    - a. 
    
    - b. 
    
    - c. 

## To Recreate my project

- Create your own .env file with your own username/password/host to use the get_connection function in my acquire.py in order to access the Zillow database.

- Make a copy of my acquire and prepare files to use the functions within the files.

- Make a copy of my final notebook, run each cell, and adjust any parameters as desired.