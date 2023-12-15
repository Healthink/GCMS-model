# GCMS-model
A machine learning model for the urine GC-MS interpretation

Background:
IEMs are increasingly detected using GC–MS urinary metabolomic analysis since it is noninvasive, effective, and affordable. However, due to the variability of the compounds excreted, interpreting urinary organic acid profiles can be challenging. For conditions where the excretion of diagnostic metabolites is a reflection of the residual activity of the faulty enzyme, the dietary load of precursors, the anabolic/catabolic status of the patient, or age, informative profiles may not always be obtained. In fact, in some "non-excretory" IEM patients, substances that are ordinarily expelled in substantial amounts may also show up at only slightly elevated or even normal concentrations. When a patient is clinically well or under appropriate nutritional control, this is especially true. Additionally, other more common clinical conditions such as "liver injury" may result in altered urinary metabolic profiles. Instead of focusing on specific anomalies, the clinically significant interpretation of organic acid values should be based on the overall pattern of metabolites present in abnormal quantities. Although comprehensive tables to aid in the interpretation of findings related to organic acids are available in published literature and online, there may still be significant uncertainty regarding the source and/or importance of a specific compound. Under acute and asymptomatic clinical states, the diagnostic sensitivity of organic acid analysis may vary significantly. Findings on organic acids can be compared to or verified by results from other analyses to arrive at a diagnosis. As a result, it can be difficult, time-consuming, and complex to analyze GC–MS urine organic acid profiles, and it demands a high level of experience and skill. These limitations also limit the ability of urine GC‒MS to be used more widely for IEM screening. 
Given the increasingly widespread implementation of GC‒MS technology to detect IEMs based on urine sample and its unique advantages, such as its noninvasive nature, efficiency and low cost compared with other technologies, the use of machine learning techniques to improve the efficiency and accuracy of urine GC‒MS data interpretation and reduce its reliance on experts and experience will greatly facilitate the wider use of this technology. The goal of this study is to utilize urine GC-MS datasets accumulated  to train and validate whether highly sensitive machine learning models can be constructed on data with such extreme imbalances in classification.

Methods:
A random forest-based machine learning model was proposed, trained and evaluated. Weighted undersampling or oversampling data processing and staged modeling strategies were used to handle the highly imbalanced data and improve the ability of the model to identify different types of rare IEM cases.

The scripts in this repository shown how the sampling and learning strategy works.










