# kc_house_project
This is a Data Science project aimed at deriving insights from data analysis to solve business problems. This project is inspired in the **`House Rocket Project`** from the Comunidade DS: https://www.comunidadedatascience.com/os-5-projetos-de-data-science-que-fara-o-recrutador-olhar-para-voce/ <br>

## 1. Business problem 
The KC Real Estate Company is a fictious company located in the King County, Washington, USA. It has a “wholetailing” business model, that is, it buys discounted properties and sells them near their retail value after making cosmetic improvements.

The business problem is to maximize profits by identifying the best opportunities in the market. In specific, the problem is to identify the properties below the market price in which the estimated improvement costs remain within a satisfactory level.

There are thousands of properties for sale with different attributes such as condition, grade, number of bedrooms, and so on. An audit identified several inefficiencies and recommended the KC company to hire a data scientist to improve the processes.

The Data Scientist was requested to: 
- Identify the best months for buying and selling properties
- Identify the most discounted properties and estimate the investment (price + estimated improvement costs) based on objective criteria (see below)
- Produce an online dashboard for the properties with buy recommendation, including a table with the recommended properties, interactive filters according to properties characteristics and a map with their locations.

The business team informed that improvement costs should not exceed 2% of the property price, and that the average improvement costs vary according to the property condition:<br>

- US$ 3.00 per sqft: condition ratings <= 2
- US$ 2.00 per sqft: condition ratings 3 or 4
- US$ 1.50 per sqft: condition rating 5

**The dataset:** https://www.kaggle.com/datasets/shivachandel/kc-house-data

## 2. Assumptions

- The KC Company is focused on properties up to US$ 800,000.00
- The `kc_house_data.csv` is the portfolio of properties available in the market
- The following columns will not be considered: `view`, `sqft_living15`, `sqft_lot15`

## 3. Solution strategy
This project was developed following these steps:<br>
- ETL: the dataset was extracted, the data consistency was analyzed and the appropriate transformations were made, including those imposed by the project’s assumptions.<br>
- EDA: exploratory analyses revealed the distribution of properties’ characteristics and price distributions.<br> 
- Insights: some hypotheses with actionable business implications were evaluated.<br>
- Algorithm: an algorithm was developed for selecting properties with low-season prices and below the 30th percentile for each region. The algorithm also estimated the improvement costs and excluded the proprieties in which the costs would represent more than 2% of the price. <br>
- Evaluation: the financial implications and the characteristics of the resulting portfolio were evaluated.
- Dashboard development: I've developed an online interactive dashboard using streamlit and a cloud server to display the properties with buy recommedantions, with the following sections: Dataset overview, Price distribution, Estimated investment (overall and segmented), and map plots. The dashboard is responsive to the filters available in the sidebar.<br> 

Link to the application: https://kc-company-dashboard.herokuapp.com <br>
This repository contains the jupyter notebook of this project.

## 4. Main insights
- The median prices are 9% higher from April to June (spring/summer)in comparison to January (winter).
- The best months for buying properties range from November to February.
- The median prices of waterfront properties are 44% higher than non-waterfront ones. 
- The cheapest properties (prices below 1st quartile) and the most expensive ones (prices above 4th quartile) have the same median condition.
- Properties that have been renovated in the past 15 years are 15.7% more expensive than non-renovated properties, and 6.7% more expensive than those renovated more than 15 years ago.
- Newer properties (yr_built above 4th quartile) are only 4.7% more expensive than older properties (yr_built below 1st quartile), but they are 15.4% more expensive than intermediate-aged properties.


## 5. Financial implications

- The budget for buying the selected properties is 27% cheaper than for buying the properties with median prices.
- The estimated impact of improvement costs is 0.87%, which is 56.5% below the cutoff value of 2%.

## 6. Concluding remarks
- The solution produced a diversified portfolio of properties with buy recommendation.
- The solution identified properties below the median market price and with satisfactory improvement costs.
- The solution produced a responsive dashboard that may be used for decision making.

## 7. Next steps

- To develop a model for predicting properties prices based on their characteristics.
- To apply this model for identifying the properties with prices much below those predicted by the model, allowing the identification of price bargains and most likely to have higher return over investment. 

## 8. Lessons learned

- To address business problems using data analysis for deriving actionable insights. <br>
- To build an app using streamlit and to deploy it in a cloud service (heroku) using git. <br>
- To develop interactive dashboards and geographic plots using geopandas, plotly and streamlit.
