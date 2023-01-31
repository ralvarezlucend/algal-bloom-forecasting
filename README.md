# Algal Bloom Forecasting in a Classification and Regression Setting
## Implementing a UNet Architecture to evaluate the differences between both settings.
This is the code repository as part of the 2022-2023 Q2 edition of the Research Project of TU Delft.

### Abstract
Forecasting algal blooms using remote sensing data is less labour-intensive and has better cover- age in time and space than direct water sampling. The paper implements a deep learning technique, the UNet Architecture, to predict the chlorophyll concentration, which is a good indicator for al- gal bloom in the Rio Negro water reservoirs of Uruguay. The research question focuses on the dif- ferences between classification and regression in algal bloom forecasting. The experiments show that the regression implementation achieves bet- ter accuracy and lower mean squared error than the classification implementation that uses cross- entropy loss and four pre-fixed bins. Different loss functions that account for the class imbalance in the data do not improve the model’s performance. Fi- nally, a quantile-based binning strategy that consid- ers the data’s underlying distribution achieves the highest accuracy in both settings.

The Research paper can be found at: http://resolver.tudelft.nl/uuid:05505158-1c11-4c3c-820e-2ade68ba753a
