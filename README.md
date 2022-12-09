# iliwycmbd_ml_challenge

***
## Team Summary 


### Features

### Model

### Model performance

### Recommendations


***
# Individual Work

## Soren Laney


### Features
|idx|	Feature Name	                       |Description                                                        | 
|---|--------------------------------------|-------------------------------------------------------------------|
|1  |tractcode                             | Tractcode of the Feature                                          | 
|2  |state                                 | State of the the Feature                                          |
|3  |county                                | County of the Feature                                             | 
|4  |isev                                  | Target                                                            | 
|5  |weighted_median_income                | Media Income Weight by Census Block Group                         | 
|6  |population_per_gas_station            | Population density ratio between Gas Stations and Census Tract    |
|7  |square_miles                          | Square Miles per Census Tract                                     |
|8  |pop_density                           | Population Density per Census Tract by Square Miles               | 
|9  |gas_stations_per_tract                | Number of Gast Stations per Census Tract                          |  
|10 |total_starbucks_by_tract              | Number of Starbucks per Census                                    |
|11 |Average_Years_of_Edcuation_by_tract   | Average Years of Education of the Population of a Cesnus Tract    |
|12 |Number_of_Schools                     | Number of Schools per Census Tracts                               |
|13 |land_m                                | Number of Square Meters in a Census Tract                         | 
|14 |nearest_big_city_pop                  | Population of the Nearest Metropolitan area of a Census Tract     |   
|15 |walmart_count                         | Number of Walmarts per Census Tract                               |
|16 |total_buildings                       | Total Number of Buildings per Census Tract                        |
|17 |hotel_count                           | Number of Hotels per Census Tract                                 |

### Model

I tried 3 different models before settling on on a gradient boosted tree. Here is some example code of those models of working on those models.

```
# Train models.

dtc = DecisionTreeClassifier(labelCol = "isev_indexed",
                            featuresCol = "features")

rf = RandomForestClassifier(labelCol = "isev_indexed",
                            featuresCol = "features",
                            numTrees = 15)

gbt = GBTClassifier(labelCol = "isev_indexed",
                            featuresCol = "features")



# Chain indexers, vectorizers, and random forest in a Pipeline
pipeline_dtc = Pipeline(stages = [labelToIndex, vector_assembler, dtc])
pipeline_rf = Pipeline(stages = [labelToIndex, vector_assembler, rf])
pipeline_gbt = Pipeline(stages = [labelToIndex, vector_assembler, gbt])

# Run each step in the pipeline.
# Train model. This also runs the indexers.
model_dtc = pipeline_dtc.fit(trainingData)
model_rf = pipeline_rf.fit(trainingData)
model_gbt = pipeline_gbt.fit(trainingData)

```


### Model performance
|Model Evaluator|Value              |
|---------------|-------------------|
|Accuracy(%)    |76.27%             |
|Test Error:    |23.72%             |     

#### Feature Importance

We can see the top ten features of the ranked from top to bottom in the graph below. We can see that most important feature, the number of hotels, has a lot of weight in detiriming whether or not a tract has an EV charger in it. This most likely because many EV companies. that build these chargers at or near hotels to accomedate for those who want to take their electric vehicle arcoss the country on a multi-day trip. We can then see that the total number of starbucks is the next largest indicator on whether or not an EV charger is in that track, contributing to 18% of feature importance in the model.

|idx|	name	                               |Feature Importance   |    
|---|--------------------------------------|---------------------|
|12	|hotel_count	                         |0.559215             |
|5	|total_starbucks_by_tract	             |0.185786             |
|6	|Average_Years_of_Edcuation_by_tract	 |0.109546             |
|1	|population_per_gas_station	           |0.041667             |
|3	|pop_density	                         |0.025232             |
|0	|weighted_median_income	               |0.025042             |
|2	|square_miles	                         |0.019208             |
|10	|walmart_count	                       |0.010639             | 
|7	|Number_of_Schools	                   |0.009559             | 
|8	|land_m	                               |0.007447             |





### Recommendations


| id|state|  tractcode|prediction|county|EV Charger Probability|
|---|-----|-----------|----------|------|----------------------|
|  0|   08|08089968400|       0.0|   089|    0.9379185479588517|
|  1|   08|08101980100|       0.0|   101|    0.9371806756432629|
|  2|   08|08001988700|       0.0|   001|    0.9334525103837586|
|  3|   08|08099000700|       0.0|   099|    0.9329660389158686|
|  4|   08|08029964800|       0.0|   029|    0.9329660389158686|
|  5|   08|08043979400|       0.0|   043|    0.9323312251892427|
|  6|   08|08101003200|       0.0|   101|    0.9318619279678754|
|  7|   08|08087000800|       0.0|   087|    0.9308353157372441|
|  8|   08|08037000100|       0.0|   037|    0.9295155114890457|
|  9|   08|08085966503|       0.0|   085|     0.927483828142378|

***

## Kavyn Abel

### Features
taxi, weighted_median_income, people_travel_to_work,pop_density, work_travel_perc, gas_stations_per_tract, median_gas_dwell_time, count_other_tracts, Average_Years_of_Edcuation_by_tract, count_fam_homes, count_non_fam_homes, avg_gas_station_visits_on_weekday, avg_gas_station_visits_on_weekend, land_m, hotel_count, walmart_count, "hour_dwell_pks, percent_carpoolers, percent_drive, extended_stay_buildings, total_buildings

### Model
GBTClassifier

### Model performance
On testing data (CA, TX, NY) 75% accuracy

On FL and MA holdout data 71% accuracy

### Recommendations
Here are the tracts in Colorado I recommend:
- 08031002701
- 08117000402
- 08069001605
- 08041001500
- 08031000303
- 08035014123
- 08067970703
- 08107000700
- 08031002803
- 08069001007

***

## Emma Holt 

### Features

['population_per_gas_station', 'weighted_median_income', 'car_truck_or_van_alone', 'avg_education_yrs', 'gas_station_percent_change', 'Number_of_Schools', 'count_fam_homes', 'land_m', 'people_travel_to_work', 'total_tract_population', 'median_gas_dwell_time', 'total_visit_counts', "hotel_count", "total_starbucks_by_tract"]

|idx|	Feature Name	                       |Description                                                        | 
|---|--------------------------------------|-------------------------------------------------------------------|
|1  |weighted_median_income                | Media Income Weight by Census Block Group                         | 
|2  |population_per_gas_station            | Population density ratio between Gas Stations and Census Tract    |
|3  |gas_stations_per_tract                | Number of Gast Stations per Census Tract                          |  
|4  |total_starbucks_by_tract              | Number of Starbucks per Census                                    |
|5  |Average_Years_of_Edcuation_by_tract   | Average Years of Education of the Population of a Cesnus Tract    |
|6  |land_m                                | Number of Square Meters in a Census Tract                         |
|7  |hotel_count                           | Number of Hotels per Census Tract                                 |
|8  |car_truck_or_van_alone                |                                                                   |
|9  |gas_station_percent_change            |                                                                   |
|10 |number_of_schools                     |                                                                   |
|11 |count_fam_homes                       |                                                                   |
|12 |people_travel_to_work                 |                                                                   |

### Model

RandomForestClassifier

### Model performance

#### Test Data
Accuracy(%): 0.7946615299944965
Test Error:  0.205338

#### Holdout states (FL, MA)
Accuracy(%): 0.7590082054941134
Test Error:  0.240992

### Recommendations

+----------+-----------+-----------------------------------------+
|prediction|tractcode  |prob_v                                   |
+----------+-----------+-----------------------------------------+
|1.0       |08041007000|[0.3059793866696401, 0.6940206133303599] |
|1.0       |08059012022|[0.3059793866696401, 0.6940206133303599] |
|1.0       |08001060200|[0.3198509873686621, 0.6801490126313379] |
|1.0       |08005080400|[0.3198509873686621, 0.6801490126313379] |
|1.0       |08041003308|[0.3198509873686621, 0.6801490126313379] |
|1.0       |08035014123|[0.341061862456565, 0.658938137543435]   |
|1.0       |08069001601|[0.3420073138748238, 0.6579926861251763] |
|1.0       |08013013212|[0.35493346315558705, 0.645066536844413] |
|1.0       |08041008000|[0.37263642192124763, 0.6273635780787524]|
|1.0       |08041002900|[0.37545206836345607, 0.6245479316365439]|
+----------+-----------+-----------------------------------------+

***

## Trey Lusk 

### Features

### Model

### Model performance

### Recommendations

***

## Paul Weinberg 

### Features

### Model

### Model performance

### Recommendations

***

## Youngwho Park

### Features

### Model

### Model performance

### Recommendations

***

## References 


