import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from string import ascii_letters
import statistics
from statistics import StatisticsError
from sklearn import datasets, linear_model
import math

#os.chdir("E:\\OneDrive\\Hult\\Data Science Intro to Python\\Assignment\\Group Assignment")

file = "world_data.xlsx"

worldOrgnl = pd.read_excel(file)

worldOrgnl = pd.concat([worldOrgnl, pd.get_dummies(worldOrgnl.income_group)], axis = 1)

droppedCol = ['country_index', 'Hult_Team_Regions', 'country_name', 'country_code', 'income_group']

worldBank = worldOrgnl.copy()
worldBankDropped = worldBank.copy()

europe = worldBankDropped.loc[worldOrgnl['Hult_Team_Regions'] == 'Europe'].copy()
europeDropped = europe.copy()
europeImputed = europeDropped.copy()

worldBank_dropped = worldBank.dropna()


## Calculte average

world = worldOrgnl.copy()
print(world.columns)

dfAvg = pd.DataFrame(columns = ['country_index', 'Hult_Team_Regions', 'country_name', 'country_code',
       'income_group', 'access_to_electricity_pop',
       'access_to_electricity_rural', 'access_to_electricity_urban',
       'CO2_emissions_per_capita', 'compulsory_edu_yrs',
       'pct_female_employment', 'pct_male_employment',
       'pct_agriculture_employment', 'pct_industry_employment',
       'pct_services_employment', 'exports_pct_gdp', 'fdi_pct_gdp', 'gdp_usd',
       'gdp_growth_pct', 'incidence_hiv', 'internet_usage_pct',
       'homicides_per_100k', 'adult_literacy_pct', 'child_mortality_per_1k',
       'avg_air_pollution', 'women_in_parliament', 'tax_revenue_pct_gdp',
       'unemployment_pct', 'urban_population_pct',
       'urban_population_growth_pct', 'High income', 'Low income',
       'Lower middle income', 'Upper middle income'])

regions = world.loc[:, 'Hult_Team_Regions']
#print(type(regions))
unique = list(set(regions))
#print(unique)

regions = ['Middle East & North Africa', 'Central Africa 2', 'Southern Latin America / Caribbean', 
        'Europe', 'World', 'Nothern Asia and Northern Pacific', 'Greater Mediteranian Region', 'Northern Europe and Northern Americas',
         'Southern Africa', 'Central Aftica 1', 'Central Asia and some Europe', 'Southern Asia and Southern Pacific', 'Northern Latin America / Caribbean']

cols = len(world.columns)
#print(len(regions))
dfAvg = pd.DataFrame(index = range(len(regions)), columns = range(cols))
#print("dfAvg Shape: ", dfAvg.shape)

dfAvg.columns = ['country_index', 'Hult_Team_Regions', 'country_name', 'country_code',
       'income_group', 'access_to_electricity_pop',
       'access_to_electricity_rural', 'access_to_electricity_urban',
       'CO2_emissions_per_capita', 'compulsory_edu_yrs',
       'pct_female_employment', 'pct_male_employment',
       'pct_agriculture_employment', 'pct_industry_employment',
       'pct_services_employment', 'exports_pct_gdp', 'fdi_pct_gdp', 'gdp_usd',
       'gdp_growth_pct', 'incidence_hiv', 'internet_usage_pct',
       'homicides_per_100k', 'adult_literacy_pct', 'child_mortality_per_1k',
       'avg_air_pollution', 'women_in_parliament', 'tax_revenue_pct_gdp',
       'unemployment_pct', 'urban_population_pct',
       'urban_population_growth_pct', 'High income', 'Low income',
       'Lower middle income', 'Upper middle income']

dfAvg.index = regions

row = -1
for region in regions:
    row = row + 1
    colNo = -1
    for col in world:
        colNo = colNo + 1
        data = world[world['Hult_Team_Regions'] == region].loc[:, col]
        
        average = 0
        try:
            average = data.mean()
        except:
            average = 0
        dfAvg.iloc[row, colNo] = average


## Average caculated



# Drop all the text columns
worldBankDropped = worldBankDropped.copy().drop(droppedCol, axis = 1)
europeDropped = europeDropped.copy().drop(droppedCol, axis = 1)
europeImputed = europeImputed.copy().drop(droppedCol, axis = 1)


#print("World Data Length: ", len(worldBank_dropped))

lenWorldBank = len(worldBank)
#print(europe['adult_literacy_pct'])
for col in worldBankDropped:
    lenTemp = len(worldBank[col].dropna())
    #print("NA Data persent in ", col, ": ", ((lenWorldBank - lenTemp)/lenWorldBank*100))
    if ((lenWorldBank - lenTemp)/lenWorldBank*100) > 20:
        droppedCol.append(col)
        worldBankDropped =  worldBankDropped.drop([col], axis = 1).copy()
        europeDropped = europeDropped.drop([col], axis = 1).copy()        
    else:
        tempCol = worldBank[col].copy().dropna()
        tempColEurope = europeDropped[col].copy().dropna()
        #print("Mean: ", mean_t)    
        if((worldBank[col].dtypes == 'int64') or (worldBank[col].dtypes == 'float64')):
            try:
                fill_value = statistics.median(tempCol)
                fill_value_Europe = statistics.median(tempColEurope)

            except StatisticsError:
                print ("No unique mode found")
                fill_value = tempCol.mean()
                fill_value_Europe = tempColEurope.mean()
            #print("Fill Value for ", col, ": ", fill_value)
            worldBankDropped[col] = worldBankDropped[col].fillna(fill_value).copy()
            europeDropped[col] = europeDropped[col].fillna(fill_value).copy()
#print("NA values in WorldBank: ", worldBankDropped.isna().any())
#print("NA values in Europe: ", europeDropped.isna().any())
regr = linear_model.LinearRegression()
colNo = -1
#print("EuropeImputed Head")
#print(europeImputed.head(15))
dfDropped = pd.DataFrame()

for col in europeImputed:
    colNo += 1
    if(col not in droppedCol):
        x = worldBankDropped.copy().drop([col], axis = 1).values.reshape(len(worldBankDropped), len(worldBankDropped.columns) -1 )
        y = worldBankDropped.loc[:, col].values.reshape(-1, 1)
        if(europeImputed[col].isna().any()):
            regr.fit(x, y)
            for i in range(0, len(europeImputed[col])- 1):                            
                if(pd.isnull(europeImputed.iloc[i, colNo])):
                    #print(europeDropped.isna().any())
                    #print(europeImputed.iloc[i, :])
                    print((regr.predict(europeDropped.iloc[i, :].copy().drop([col], axis = 0).values.reshape(1, -1)))[0][0])
                    europeImputed.iloc[i, colNo] = (regr.predict(europeDropped.iloc[i, :].copy().drop([col], axis = 0).values.reshape(1, -1)))[0][0]
                    
                    #print(europeDropped.iloc[i, colNo])

for col in droppedCol:    
    if(col in europeImputed.columns):        
        europeImputed.drop(col, axis = 1)
    if(europe[col].isna().sum() <= europe[col].size/2):
        tempCol_dropped = europe[col].copy()
        tempCol_filled = tempCol_dropped.copy()
        tempCol_dropped = tempCol_dropped.dropna()
        ###########################
        if((europe[col].dtypes == 'int64') or (europe[col].dtypes == 'float64')):
            try:
                fill_value = statistics.median(tempCol_dropped)                

            except StatisticsError:
                print ("No unique mode found")
                fill_value = tempCol_dropped.mean()
                
            #print("Fill Value for ", col, ": ", fill_value)
            tempCol_filled = tempCol_filled.fillna(fill_value).copy()
            
        ###########################
        dfDropped = pd.concat([dfDropped, tempCol_filled], axis = 1)
    else:
        dfDropped = pd.concat([dfDropped, europe[col]], axis = 1)
#print(dfDropped)

dfWorldImputed = pd.concat([dfDropped, europeImputed], axis = 1)
writer = pd.ExcelWriter('EuropeImputed.xlsx')
dfWorldImputed.to_excel(writer,'ImputedData')


outlier = pd.DataFrame(columns = ['Index', 'Upper', 'Lower'])
#print(dfWorldImputed.head(15))
input("Enter:")
for col in range(0, len(dfWorldImputed.columns)):    
    if( (dfWorldImputed.iloc[:, col].dtypes == 'int64') or (dfWorldImputed.iloc[:, col].dtypes == 'float64')   ):
        
        indexOutlier = dfWorldImputed.columns[col]
        #print(indexOutlier)
        #outlier.iloc[col, 0] = dfWorldImputed.columns[col]
        q = dfWorldImputed.iloc[:, col].quantile(0.98)
        U = dfWorldImputed[dfWorldImputed.iloc[:, col] > q].iloc[:, 2].values
        #print(type(U))
        print(U)
        #print(U[0].iloc[:, 2].values)
        #dfWorldImputed.iloc[len(dfWorldImputed)+1, col] = U[0].iloc[:, 2].values

        q = dfWorldImputed.iloc[:, col].quantile(0.1)
        L = dfWorldImputed[dfWorldImputed.iloc[:, col] < q].iloc[:, 2].values
        #print(L[0].iloc[:, 2].values)
        #dfWorldImputed.iloc[len(dfWorldImputed)+1, col] = L[0].iloc[:, 2].values
        newRow = [indexOutlier, U, L]
        outlier.loc[len(outlier)] = newRow
        #print(q)

        #print(dfWorldImputed.dtypes.index[colNo])
        #plt.boxplot(dfWorldImputed.iloc[:, col], patch_artist = True, xlabel = dfWorldImputed.dtypes.index[colNo])
        
        
        fig = plt.figure()
        fig.suptitle(dfWorldImputed.dtypes.index[col], fontsize=14, fontweight='bold')

        ax = fig.add_subplot(111)
        ax.boxplot(dfWorldImputed.iloc[:, col])

        ax.set_title(dfWorldImputed.iloc[:, col])
        for vals in dfAvg[dfWorldImputed.dtypes.index[col]]:
            ax.axhline(y = vals, label = dfWorldImputed.dtypes.index[col], c = 'b', linestyle = ':')
        ax.axhline(y = vals, label = "World Average", c = 'r', linestyle = '-.', linewidth = 3)


        #strOutlier = "Outliers: " + L + " " + U
        #ax.set_xlabel(strOutlier)
        #ax.set_ylabel('ylabel')
        #ax.set_title('axes title')
        #plt.show()
        savefile = dfWorldImputed.dtypes.index[col] + ".png"
        fig.savefig(savefile)
        plt.close(fig)


outlier.to_excel(writer,'outlier')





       
corr = europeImputed.corr()
corr.to_excel(writer,'Corr')
writer.save()

#plt.matshow(europe.corr())
#plt.show()
#print(corr)
sns.set(style = 'white')
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(40, 26)),
                 columns=list(ascii_letters[26:]))

#print(corr.iloc[2, 1])

#Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

##############################################################################
##############################################################################
######## From JUan
##############################################################################
##############################################################################




world_data = pd.read_excel(file)


###############################################################################
# Calculate amount of missing values in the dataset
###############################################################################
column_names = list(world_data.columns.values)
total_obs = world_data.shape[0]
total_columns = world_data.shape[1]
missing_values = []
missing_values_pct = []

for column in world_data:
    missing_values.append(total_obs - world_data[column].describe()[0])
    missing_values_pct.append((total_obs - world_data[column].describe()[0])/total_obs)
    

dictionary = {'column':column_names ,'missing_values':missing_values,'missing_values_pct':missing_values_pct} 
missing_values_df = pd.DataFrame(dictionary)    

    
###############################################################################
# Calculate mean, median and standard deviation for non missing values
###############################################################################
    
income_group = ['High income','Upper middle income','Lower middle income', 'Low income']
data_columns = column_names[5:]

mean_1 = pd.DataFrame()
median_2 = pd.DataFrame()
std_3 = pd.DataFrame()
max_4 = pd.DataFrame()
min_5 = pd.DataFrame()

df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()

list1 = []
list2 = []
list3 = []
list4 = []
list5 = []

dict1 = {}
dict2 = {}
dict3 = {}
dict4 = {}
dict5 = {}

for i in data_columns:
    for item in income_group:
        
        list1.append(world_data[world_data['income_group'] == item][i].mean(skipna=True))
        list2.append(world_data[world_data['income_group'] == item][i].median(skipna=True))
        list3.append(world_data[world_data['income_group'] == item][i].std(skipna=True))
        list4.append(world_data[world_data['income_group'] == item][i].max(skipna=True))
        list5.append(world_data[world_data['income_group'] == item][i].min(skipna=True))
        
        dict1 = {i:list1}
        dict2 = {i:list2}
        dict3 = {i:list3}
        dict4 = {i:list4}
        dict5 = {i:list5}
    
    df1 = pd.DataFrame(dict1)
    df2 = pd.DataFrame(dict2)
    df3 = pd.DataFrame(dict3)
    df4 = pd.DataFrame(dict4)
    df5 = pd.DataFrame(dict5)
    
    mean_1 = pd.concat([mean_1,df1],axis=1)
    median_2 = pd.concat([median_2,df2],axis=1)
    std_3 = pd.concat([std_3,df3],axis=1)
    max_4 = pd.concat([max_4,df4],axis=1)
    min_5 = pd.concat([min_5,df5],axis=1)
    
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    
    dict1 = {}
    dict2 = {}
    dict3 = {}
    dict4 = {}
    dict5 = {}

###############################################################################
#subsetting world data by Hult Team Region equal to europe 
###############################################################################

data = world_data[world_data['Hult_Team_Regions'] == 'Europe']

total_obs_europe = data.shape[0]
missing_values_europe = []
missing_values_pct_europe = []
list= []
dictionary= {}

for column in data:
    missing_values_europe.append(total_obs_europe - data[column].describe()[0])
    missing_values_pct_europe.append((total_obs_europe - data[column].describe()[0])/total_obs_europe)

dictionary = {'column':column_names ,'missing_values':missing_values_europe,'missing_values_pct':missing_values_pct_europe} 
missing_values_df_europe = pd.DataFrame(dictionary)   

list = missing_values_europe[0:31]
col_missing=0
col_ok = 0
for i in list:
    if i>0:
        col_missing +=1
    else:
        col_ok += 1

total_missing_values = sum(missing_values_df['missing_values'])
total_missing_values_europe = sum(list)

###############################################################################
# Mean or median?
###############################################################################

p=0
list_bool= []
list_value = []
list_impute = []
df_bool = pd.DataFrame()
df_value = pd.DataFrame()
df_value_impute = pd.DataFrame()
df_bool_temp = pd.DataFrame()
df_value_temp = pd.DataFrame()
df_value_impute_temp = pd.DataFrame()


for item in data_columns:
    for i in range(0,4):
        list1 = mean_1.loc[i,item]
        mn = pd.to_numeric(list1)
        list2 = median_2.loc[i,item]
        md = pd.to_numeric(list2)
        list3 = std_3.loc[i,item]
        sd = pd.to_numeric(list3)
        list_bool.append(abs(mn-md)<sd)
        list_value.append(sd - abs(mn-md))
        if abs(mn-md)<sd == True:
            list_impute.append(mn)
        else:
            list_impute.append(md)
        
        p +=1
    
      
    dict1 = {item:list_bool}
    dict2 = {item:list_value}
    dict3 = {item:list_impute}
    
    df_bool_temp = pd.DataFrame(dict1)
    df_value_temp = pd.DataFrame(dict2)
    df_value_impute_temp = pd.DataFrame(dict3)
    
    df_bool = pd.concat([df_bool,df_bool_temp],axis=1)
    df_value = pd.concat([df_value,df_value_temp],axis=1)
    df_value_impute = pd.concat([df_value_impute,df_value_impute_temp],axis=1)
    
    list_bool =[]
    list_value = []
    list_impute=[]
    
print(p)

###############################################################################
# Flag missing values
###############################################################################

for col in world_data:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if world_data[col].isnull().any():
        world_data['m_'+col] = world_data[col].isnull().astype(int)


data_europe = world_data[world_data['Hult_Team_Regions'] == 'Europe']  
       
   
###############################################################################
# Histograms
###############################################################################




for item in data_columns:
        
        a = world_data[world_data['income_group'] == 'High income'][item].dropna()
        plt.subplot(2, 2, 1)
        plt.hist(a, bins = 'fd', color='blue', alpha = 0.3)
        plt.title('High income')  
        list1 = mean_1.loc[0, item]
        mn = pd.to_numeric(list1)
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        list2 = median_2.loc[0, item]
        md = pd.to_numeric(list2)
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        
        b = world_data[world_data['income_group'] == 'Upper middle income'][item].dropna()
        plt.subplot(2, 2, 2)
        plt.hist(b, bins = 'fd', color='green', alpha = 0.3)
        plt.title('Upper middle income')
        list1 = mean_1.loc[1, item]
        mn = pd.to_numeric(list1)
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        list2 = median_2.loc[1, item]
        md = pd.to_numeric(list2)
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        c = world_data[world_data['income_group'] == 'Lower middle income'][item].dropna()
        plt.subplot(2, 2, 3) 
        plt.hist(c, bins = 'fd', color='red', alpha = 0.3)
        plt.xlabel('Lower middle income')
        list1 = mean_1.loc[2, item]
        mn = pd.to_numeric(list1)
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        list2 = median_2.loc[2, item]
        md = pd.to_numeric(list2)
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        d = world_data[world_data['income_group'] == 'Low income'][item].dropna()
        plt.subplot(2, 2, 4)
        plt.hist(d, bins = 3, color='purple', alpha = 0.1)
        plt.xlabel('Low income')
        list1 = mean_1.loc[2, item]
        mn = pd.to_numeric(list1)
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        list2 = median_2.loc[2, item]
        md = pd.to_numeric(list2)
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        d = 'Histogram_'+item+' before imputation.png'
        
        plt.savefig(d)
        plt.show()


###############################################################################
# Value imputation
###############################################################################


list1=[]
list2=[]
list3=[]
list4=[]
list_index = data_europe.index
column = column_names[5:]
        
for item in column:
    for i in list_index :
        if data_europe.loc[i,'income_group'] == 'High income':
            list1 = df_value_impute.loc[0,item]
            value = pd.to_numeric(list1)
            data_europe[item] = data_europe[item].fillna(value).astype(float)
         
        elif data_europe.loc[i,'income_group'] == 'Upper middle income':
            list2 = df_value_impute.loc[1,item]
            value = pd.to_numeric(list2)
            data_europe[item] = data_europe[item].fillna(value).astype(float)
    
        elif data_europe.loc[i,'income_group'] == 'Lower middle income':
            list3 = df_value_impute.loc[2,item]
            value = pd.to_numeric(list3)
            data_europe[item] = data_europe[item].fillna(value).astype(float)
    
        elif data_europe.loc[i,'income_group'] == 'Low income':
            list4 = df_value_impute.loc[3,item]
            value = pd.to_numeric(list4)
            data_europe[item] = data_europe[item].fillna(value).astype(float)
        else:
            print('error')
    
           
###############################################################################
# Histograms
###############################################################################




for item in data_columns:
        
        a = data_europe[data_europe['income_group'] == 'High income'][item]
        plt.subplot(2, 2, 1)
        plt.hist(a, bins = 'fd', color='blue', alpha = 0.3)
        plt.title('High income')  
        mn = data_europe[data_europe['income_group'] == 'High income'][item].mean()
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        md = data_europe[data_europe['income_group'] == 'High income'][item].median()
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        
        b = world_data[world_data['income_group'] == 'Upper middle income'][item].dropna()
        plt.subplot(2, 2, 2)
        plt.hist(b, bins = 'fd', color='green', alpha = 0.3)
        plt.title('Upper middle income')
        mn = data_europe[data_europe['income_group'] == 'Upper middle income'][item].mean()
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        md = data_europe[data_europe['income_group'] == 'Upper middle income'][item].median()
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        c = world_data[world_data['income_group'] == 'Lower middle income'][item].dropna()
        plt.subplot(2, 2, 3) 
        plt.hist(c, bins = 'fd', color='red', alpha = 0.3)
        plt.xlabel('Lower middle income')       
        mn = data_europe[data_europe['income_group'] == 'Lower middle income'][item].mean()
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        md = data_europe[data_europe['income_group'] == 'Lower middler income'][item].median()
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        d = world_data[world_data['income_group'] == 'Low income'][item].dropna()
        plt.subplot(2, 2, 4)
        plt.hist(d, bins = 3, color='purple', alpha = 0.1)
        plt.xlabel('Low income')
        mn = data_europe[data_europe['income_group'] == 'Low income'][item].mean()
        plt.axvline(x = mn, label = 'mean', linestyle = '--', color = 'r')
        md = data_europe[data_europe['income_group'] == 'Low income'][item].median()
        plt.axvline(x = md, label = 'median', linestyle = '--', color = 'b')
        
        d = 'Histogram_'+item+' after imputation.png'
        
        plt.savefig(d)
        plt.show()

###############################################################################
# Boxplot
###############################################################################



for item in data_columns:
    data_europe.boxplot(column = [item],by ='income_group' ,vert = False, manage_xticks = True , patch_artist = True, meanline = True, showmeans = True)                    
    plt.title(item + " by Channel")
    plt.suptitle("")
    box_name = 'Boxplot_'+item+' after imputation.png'
    plt.savefig(box_name)                   
    plt.show()   

data_europe.to_excel('data_europe.xlsx', index = False)

