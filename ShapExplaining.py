import shap as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


my_model = ...
X_test, y_test = ...

explainer = sp.TreeExplainer(my_model)
shap_values = explainer.shap_values(X_test, approximate=True)

exp = sp.Explanation(values=shap_values[1],
                base_values=explainer.expected_value[1],
                data=X_test,
                feature_names=X_test.columns.tolist()
                )
# summary plot
custom_cmap = mcolors.ListedColormap(sns.color_palette("Blues", 10,)[:-2])
fig = plt.figure(dpi=120)
sp.summary_plot(shap_values, X_test, title='', show=True, class_names=my_model.classes_, 
                color=custom_cmap.reversed())

# waterfall plot
target = 2 # set positive class
ptr = X_test.index.to_list().sample()
tmp = sp.Explanation(values=shap_values[target][ptr], 
                        base_values=explainer.expected_value[target], 
                        data=X_test.iloc[ptr],
                        feature_names=X_test.columns.to_list()
                        )
sp.plots.waterfall(tmp)
