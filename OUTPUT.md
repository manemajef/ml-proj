# Output from the notebook (`explore.py`)


```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 63464 entries, 0 to 63463
Data columns (total 29 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   Client_ID                    63464 non-null  int64  
 1   Professionals_Count          63464 non-null  int64  
 2   Students_Count               63460 non-null  float64
 3   Observers_Count              63464 non-null  int64  
 4   Course_Start_Date            63464 non-null  object 
 5   Practical_Hours              63464 non-null  int64  
 6   Theory_Hours                 63464 non-null  int64  
 7   Registration_Days_Before     60798 non-null  float64
 8   Origin_Country               62907 non-null  object 
 9   Catering_Package             63057 non-null  object 
 10  Welcome_Gift_Type            63464 non-null  object 
 11  Requested_Lab_Config         61728 non-null  object 
 12  Assigned_Lab_Config          63464 non-null  object 
 13  Prev_Course_Dropouts         63464 non-null  int64  
 14  Prev_Course_Attended         63464 non-null  int64  
 15  Pre_Course_Supports_Tickets  63464 non-null  int64  
 16  Physical_Course_Kits         62424 non-null  float64
 17  Waiting_List_Days            63464 non-null  int64  
 18  Registration_Changes         63464 non-null  int64  
 19  Enrollment_Type              62745 non-null  object 
 20  Lanyard_Color                63464 non-null  object 
 21  Client_Category              63464 non-null  object 
 22  Submission_Source            62859 non-null  object 
 23  Returning_Client             63464 non-null  int64  
 24  Agent_ID                     52291 non-null  float64
 25  Company_ID                   3120 non-null   float64
 26  Payment_Terms                62877 non-null  object 
 27  Daily_Tuition_Cost           63385 non-null  float64
 28  Dropped_Course               63464 non-null  int64  
dtypes: float64(6), int64(12), object(11)
memory usage: 14.0+ MB


                          null_count  null_percent
Company_ID                     60344     95.083827
Agent_ID                       11173     17.605257
Registration_Days_Before        2666      4.200807
Requested_Lab_Config            1736      2.735409
Physical_Course_Kits            1040      1.638724
Enrollment_Type                  719      1.132926
Submission_Source                605      0.953296
Payment_Terms                    587      0.924934
Origin_Country                   557      0.877663
Catering_Package                 407      0.641308
Daily_Tuition_Cost                79      0.124480
Students_Count                     4      0.006303




              count      mean       std  min  25%  50%  75%  max
Company_ID                                                      
False        3120.0  0.212179  0.408917  0.0  0.0  0.0  0.0  1.0
True        60344.0  0.424848  0.494324  0.0  0.0  0.0  1.0  1.0
unique company_id list is too big:  True


number of agents + 1 =  204


Mean drop rate 0.4143924114458591


          count      mean        se  diff_from_global    z_score
Agent_ID                                                        
218.0      6551  0.734850  0.006086          0.320457  52.651970
104.0      2577  0.129996  0.009704         -0.284396 -29.307036
224.0      1312  0.048780  0.013600         -0.365612 -26.883018
264.0      2380  0.181092  0.010098         -0.233300 -23.104359
129.0       601  0.800333  0.020094          0.385940  19.206499
139.0       744  0.737903  0.018060          0.323511  17.912905
314.0       222  1.000000  0.033062          0.585608  17.712258
223.0       148  1.000000  0.040493          0.585608  14.461998
313.0       231  0.852814  0.032412          0.438421  13.526596
118.0       336  0.062500  0.026874         -0.351892 -13.093938
189.0       310  0.096774  0.027979         -0.317618 -11.352121
250.0       114  0.929825  0.046138          0.515432  11.171582
138.0       761  0.611038  0.017857          0.196646  11.012034
320.0      1071  0.577965  0.015053          0.163572  10.866627
205.0      1074  0.576350  0.015032          0.161958  10.774434
133.0       246  0.752033  0.031408          0.337640  10.750096
262.0       254  0.102362  0.030910         -0.312030 -10.094952
261.0       216  0.731481  0.033518          0.317089   9.460169
317.0       183  0.071038  0.036415         -0.343354  -9.428848
306.0       310  0.670968  0.027979          0.256575   9.170362
169.0       352  0.184659  0.026257         -0.229733  -8.749557
321.0       220  0.136364  0.033212         -0.278029  -8.371279
300.0       335  0.602985  0.026915          0.188593   7.007094
220.0       513  0.267057  0.021750         -0.147336  -6.774199
252.0       131  0.160305  0.043040         -0.254087  -5.903492
147.0       187  0.620321  0.036024          0.205928   5.716470
190.0       134  0.171642  0.042556         -0.242751  -5.704315
217.0       168  0.202381  0.038006         -0.212011  -5.578337
114.0       107  0.158879  0.047623         -0.255514  -5.365339
158.0       892  0.328475  0.016494         -0.085917  -5.208980
277.0       240  0.570833  0.031798          0.156441   4.919792
111.0       129  0.620155  0.043372          0.205763   4.744081
258.0       493  0.517241  0.022186          0.102849   4.635688
184.0     22109  0.401285  0.003313         -0.013108  -3.956465
222.0       358  0.312849  0.026036         -0.101543  -3.900168
302.0       102  0.578431  0.048776          0.164039   3.363085
167.0       246  0.516260  0.031408          0.101868   3.243359
219.0      1983  0.379728  0.011062         -0.034665  -3.133573
121.0       182  0.510989  0.036515          0.096597   2.645381
199.0       106  0.292453  0.047847         -0.121940  -2.548522
318.0       225  0.480000  0.032841          0.065608   1.997727
178.0       200  0.455000  0.034833          0.040608   1.165770
260.0       118  0.398305  0.045349         -0.016087  -0.354744
144.0       277  0.415162  0.029598          0.000770   0.026016
mean for eearly regs 0.6483718487394958
mean for missing vals 0.41522880720180044


mean for rest 0.33612781459860425


Logsitic Regression first score: 0.8987543602442913


The raw score before preprocess is:  0.9067554948413687


Random Forest Basline AOC: 0.8989351770621097

```
