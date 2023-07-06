# Benchmarks

Benchmark has been done with some tabular datasets from the [Tabular data learning benchmark](https://github.com/LeoGrin/tabular-benchmark/tree/main). It is also
hosted on [Hugging Face](https://huggingface.co/datasets/inria-soda/tabular-benchmark).

For the binary classification task, `sklearn.metrics.f1_score` is used for evaluation. For the regression task, `sklearn.metrics.mean_squared_error` is used for evaluation.

The downloaded datasets are divided into 3 sections: `train`: 50%, `val`: 10%, `test`: 40%.
Feature importance is calculated from the `train` set. Feature selection is done on the `val` set. 
The final benchmark is evaluated on the `test` set. Therefore the `test` set is unseen to both the feature importance and selection process.


Raw result data are in [`benchmarks/results`](https://github.com/kingychiu/target-permutation-importances/tree/main/benchmarks/results).


## Binary Classification Results with RandomForest

built-in: The baseline, it is the built-in importances from the model.

| dataset                                    | importances  | feature_reduction | test_f1    |
| ------------------------------------------ | ------------ | ----------------- | ---------- |
| clf_cat/electricity.csv                    | built-in     | 8->2              | 0.894      |
| clf_cat/electricity.csv                    | **A-R**      | 8->4              | **0.9034** |
| clf_cat/electricity.csv                    | A/(R+1)      | 8->2              | 0.894      |
| clf_cat/eye_movements.csv                  | built-in     | 23->22            | 0.6169     |
| clf_cat/eye_movements.csv                  | **A-R**      | 23->10            | **0.6772** |
| clf_cat/eye_movements.csv                  | A/(R+1)      | 23->22            | 0.6212     |
| clf_cat/covertype.csv                      | built-in     | 54->26            | 0.9558     |
| clf_cat/covertype.csv                      | **A-R**      | 54->52            | **0.9586** |
| clf_cat/covertype.csv                      | A/(R+1)      | 54->30            | 0.9547     |
| clf_cat/albert.csv                         | built-in     | 31->22            | 0.6518     |
| clf_cat/albert.csv                         | **A-R**      | 31->24            | **0.6587** |
| clf_cat/albert.csv                         | A/(R+1)      | 31->22            | 0.6527     |
| clf_cat/compas-two-years.csv               | built-in     | 11->10            | 0.6316     |
| clf_cat/compas-two-years.csv               | **A-R**      | 11->2             | **0.6589** |
| clf_cat/compas-two-years.csv               | A/(R+1)      | 11->6             | 0.6335     |
| clf_cat/default-of-credit-card-clients.csv | built-in     | 21->18            | 0.671      |
| clf_cat/default-of-credit-card-clients.csv | **A-R**      | 21->17            | **0.6826** |
| clf_cat/default-of-credit-card-clients.csv | A/(R+1)      | 21->20            | 0.6797     |
| clf_cat/road-safety.csv                    | **built-in** | 32->31            | **0.7895** |
| clf_cat/road-safety.csv                    | A-R          | 32->30            | 0.7886     |
| clf_cat/road-safety.csv                    | A/(R+1)      | 32->29            | 0.7893     |
| clf_num/Bioresponse.csv                    | built-in     | 419->295          | 0.7686     |
| clf_num/Bioresponse.csv                    | A-R          | 419->214          | 0.7692     |
| clf_num/Bioresponse.csv                    | **A/(R+1)**  | 419->403          | **0.775**  |
| clf_num/jannis.csv                         | built-in     | 54->22            | 0.7958     |
| clf_num/jannis.csv                         | A-R          | 54->28            | 0.7988     |
| clf_num/jannis.csv                         | **A/(R+1)**  | 54->26            | **0.7998** |
| clf_num/MiniBooNE.csv                      | built-in     | 50->33            | 0.9306     |
| clf_num/MiniBooNE.csv                      | A-R          | 50->47            | 0.93       |
| clf_num/MiniBooNE.csv                      | **A/(R+1)**  | 50->49            | **0.9316** |

## Regression Results with RandomForest

built-in: The baseline, it is the built-in importances from the model.

| dataset                                         | importances  | feature_reduction | test_mse            |
| ----------------------------------------------- | ------------ | ----------------- | ------------------- |
| reg_num/cpu_act.csv                             | built-in     | 21->20            | 6.0055              |
| reg_num/cpu_act.csv                             | A-R          | 21->20            | 6.0099              |
| reg_num/cpu_act.csv                             | **A/(R+1)**  | 21->19            | **5.9768**          |
| reg_num/pol.csv                                 | **built-in** | 26->16            | **0.2734**          |
| reg_num/pol.csv                                 | A-R          | 26->26            | 0.278               |
| reg_num/pol.csv                                 | A/(R+1)      | 26->12            | 0.2786              |
| reg_num/elevators.csv                           | built-in     | 16->7             | 8.0447              |
| reg_num/elevators.csv                           | A-R          | 16->15            | 8.3465              |
| reg_num/elevators.csv                           | **A/(R+1)**  | 16->6             | **7.8848**          |
| reg_num/wine_quality.csv                        | built-in     | 11->11            | 0.4109              |
| reg_num/wine_quality.csv                        | **A-R**      | 11->10            | **0.4089**          |
| reg_num/wine_quality.csv                        | A/(R+1)      | 11->11            | 0.4122              |
| reg_num/Ailerons.csv                            | built-in     | 33->12            | 2.8274              |
| reg_num/Ailerons.csv                            | **A-R**      | 33->29            | **2.8125**          |
| reg_num/Ailerons.csv                            | A/(R+1)      | 33->12            | 2.8304              |
| reg_num/yprop_4_1.csv                           | built-in     | 42->26            | 75403.6496          |
| reg_num/yprop_4_1.csv                           | A-R          | 42->41            | 75081.8961          |
| reg_num/yprop_4_1.csv                           | **A/(R+1)**  | 42->32            | **74671.0854**      |
| reg_num/superconduct.csv                        | built-in     | 79->53            | 54470.4924          |
| reg_num/superconduct.csv                        | **A-R**      | 79->63            | **54011.8479**      |
| reg_num/superconduct.csv                        | A/(R+1)      | 79->60            | 54454.3817          |
| reg_cat/topo_2_1.csv                            | built-in     | 255->217          | 76175.864           |
| reg_cat/topo_2_1.csv                            | A-R          | 255->254          | 76206.9714          |
| reg_cat/topo_2_1.csv                            | **A/(R+1)**  | 255->226          | **76140.8313**      |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **built-in** | 359->6            | **177937.9184**     |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | A-R          | 359->194          | 183405.9763         |
| reg_cat/Mercedes_Benz_Greener_Manufacturing.csv | **A/(R+1)**  | 359->6            | **177937.9184**     |
| reg_cat/house_sales.csv                         | **built-in** | 17->16            | **110072.8755**     |
| reg_cat/house_sales.csv                         | A-R          | 17->17            | 110141.2913         |
| reg_cat/house_sales.csv                         | A/(R+1)      | 17->17            | 110404.0862         |
| reg_cat/nyc-taxi-green-dec-2016.csv             | **built-in** | 16->15            | **10585.6377**      |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A-R          | 16->4             | 10758.4811          |
| reg_cat/nyc-taxi-green-dec-2016.csv             | A/(R+1)      | 16->15            | 10589.5054          |
| reg_cat/Allstate_Claims_Severity.csv            | **built-in** | 124->113          | **1002055785.0415** |
| reg_cat/Allstate_Claims_Severity.csv            | A-R          | 124->124          | 1003019739.9178     |
| reg_cat/Allstate_Claims_Severity.csv            | A/(R+1)      | 124->102          | 1003113924.3013     |

---

