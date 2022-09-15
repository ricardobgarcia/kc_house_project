# Developed by Ricardo B Garcia 
# for studying purposes 
# rbgarcia@gmail.com
#------------------------
# Libraries
#------------------------
import numpy as np
import pandas as pd
import geopandas
import folium
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins   import MarkerCluster
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

#------------------------
# Configs/settings
#------------------------
st.set_page_config(layout='wide')
@st.cache(allow_output_mutation=True)

#------------------------
# Functions
#------------------------

# Function to get data
def get_data( path ):
    df = pd.read_csv( path )  
    return df

# Function for loading maps
def get_geofile( url ):
    geofile = geopandas.read_file( url )
    return geofile

# Function for data transformation
def set_transformations( df ): 
    
    # Drop duplicates
    df.drop_duplicates(subset='id', keep='last', inplace=True)
    
    # Adjust data type
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    #df['date'] = df['date'].dt.date
    
    # Correcting some issues
    
    # Get the index of items with NaN in 'sqft_above'
    idxs = df.loc[df['sqft_above'].isna()].index.to_list()

    # For each item
    for i in idxs:
        # Compute the sqft_above: sqft_living - sqft_basement
        x_above = df.loc[i, 'sqft_living'] - df.loc[i, 'sqft_basement']
        # Insert the value
        df.loc[i, 'sqft_above'] = x_above
    
    # Set column type to int (it was flot due to the presence of NaN)
    df['sqft_above'] = df['sqft_above'].astype(int)
    
    # Another correction
    idx = df.loc[df['bedrooms'] == 33].index
    df.loc[idx, 'bedrooms'] = 3
    
    # This project only considers properties below 800,000
    df = df.loc[df['price'] <= 800000]

    # Droping unnecessary columns to this project
    cols_to_drop = ['view', 'sqft_living15', 'sqft_lot15']
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    # Creating features
    # Renovation
    df['renovation'] = df['yr_renovated'].apply(lambda x: 'no' if x == 0 else
                                                              'before 1995' if x < 1995 else
                                                              'after 1995')
    # Basement
    df['basement'] = df['sqft_basement'].apply(lambda x: 'yes' if x > 0 else 'no')
    
    # Waterview: changing from 0/1 to no/yes
    df['waterfront'] = df['waterfront'].apply(lambda x: 'yes' if x == 1 else 'no')

    return df

# Function for selecting properties with buy recommendation
def select_properties( dataset, percentile):
    """This function selects properties with prices below the
    informed percentile for each region (zipcode).
    dataset: the dataset of the properties
    percentile: the percentile cutoff price 
    """
    df_buy = dataset.copy() # Make a copy of the dataset
    
    # Initialize useful lists
    buying_months = [1, 2, 11, 12] 
    zipcodes = sorted(df_buy['zipcode'].unique())
    
    
    # Market time
    df_buy['market_time'] = df_buy['date'].apply(lambda x: 'buy' if x.month in buying_months
                                                            else 'sell')
    
    # Select properties announced during the buying months
    df_buy = df_buy.loc[df_buy['market_time'] == 'buy']
    
    
    # Select prices below the quantile cutoff
    for region in zipcodes:
        # Median price of the region
        price_mdn = dataset.loc[dataset['zipcode'] == region, 'price'].quantile(0.50)      
        df_buy.loc[df_buy['zipcode'] == region, 'price_mdn'] = price_mdn
         
        # Select prices below the percentile cutoff informed
        cutoff = df_buy.loc[df_buy['zipcode'] == region, 'price'].quantile(percentile)
        idx_to_drop = df_buy[(df_buy['zipcode'] == region) & (df_buy['price'] >= cutoff)].index
        df_buy.drop(idx_to_drop, inplace=True)
    
    # Estimating improvement costs
    df_buy['est_costs'] = df_buy['condition'].apply(lambda x: 3 if x <= 2 else
                                                              2 if x <= 4 else
                                                              1.5)
    
    df_buy['est_costs'] = df_buy['est_costs'] * df_buy['sqft_living']
    
    df_buy['est_investment'] = df_buy['price'] + df_buy['est_costs']
    
    df_buy['est_costs %'] = ( (df_buy['est_investment'] / df_buy['price']) -1) * 100
    
    # Filter and order by 'est_costs %'
    df_buy = df_buy.loc[df_buy['est_costs %'] <= 2]
    df_buy = df_buy.sort_values('est_costs %')
        
    return df_buy

# Function for displaying tables
def data_overview( df ):
    # Page title
    st.title('KC Real Estate Company')
    
    st.write('This application displays the properties with buy recommendation.')
    st.write('The sidebar brings filters for selecting properties based on their characteristics.')

    # Sidebar
    st.sidebar.title('Filters')

    st.sidebar.header('Location')

    # Location filter
    f_zipcode = st.sidebar.multiselect(
        'Enter zipcode',
       df['zipcode'].sort_values().unique())

    # Aplly filter here
    if (f_zipcode == []):
        df_f = df.copy()
    else:
        df_f = df.loc[df['zipcode'].isin(f_zipcode)]

    # Columns to be displayed
    cols = ['id', 'zipcode', 'waterfront', 'price', 
            'est_investment', 'est_costs', 'est_costs %', 
            'price_mdn', 'condition', 'grade', 'yr_built', 'yr_renovated', 
            'floors', 'basement', 'bedrooms', 'bathrooms', 
            'sqft_living', 'sqft_above','sqft_basement', 'lat', 'long']
    
    df_f = df_f.loc[:, cols]

    # Location waterfront
    f_waterfront = st.sidebar.checkbox('Waterfront house')
    if f_waterfront:
        df_f = df_f.loc[df['waterfront']=='yes']
    
    # condition filter
    st.sidebar.subheader('Condition')
    min_cond = int(df['condition'].min())
    max_cond = int(df['condition'].max())
    f_cond = st.sidebar.slider('Enter max condition',
                                min_cond,
                                max_cond,
                                max_cond)
    
    # grade filter
    st.sidebar.subheader('Grade')
    min_grade = int(df['grade'].min())
    max_grade = int(df['grade'].max())
    f_grade = st.sidebar.slider('Enter max grade',
                                min_grade,
                                max_grade,
                                max_grade)
    
    # est_cost% filter
    min_est_cost = float(df['est_costs %'].min())
    max_est_cost = float(df['est_costs %'].max())

    st.sidebar.subheader('Estimated costs %')
    f_est_cost = st.sidebar.slider('Enter max est_costs %',
                                min_est_cost,
                                max_est_cost,
                                max_est_cost)
    # bedrooms filter
    st.sidebar.subheader('Bedrooms')
    min_bed = int(df['bedrooms'].min())
    max_bed = int(df['bedrooms'].max())
    f_bedrooms = st.sidebar.slider('Enter max bedrooms',
                                    min_bed,
                                    max_bed,
                                    max_bed)
    
    # bathrooms filter
    st.sidebar.subheader('Bathrooms')
    max_bath = df['bathrooms'].max()
    n_bath = sorted(df['bathrooms'].unique()).index(max_bath)
    f_bathrooms = st.sidebar.selectbox('Enter max bathrooms',
                                    sorted(df['bathrooms'].unique()),
                                      n_bath)

    # floors filter
    st.sidebar.subheader('Floors')
    max_floor = df['floors'].max()
    n_floor = sorted(df['floors'].unique()).index(max_floor)
    f_floors = st.sidebar.selectbox('Enter max floors',
                                    sorted(df['floors'].unique()),
                                      n_floor)

    # Apply filters here
    df_f = df_f.loc[(df_f['condition']<=f_cond) &
                    (df_f['grade']<=f_grade) &
                    (df_f['est_costs %']<=f_est_cost) &
                    (df_f['bedrooms']<=f_bedrooms) &
                    (df_f['bathrooms']<=f_bathrooms) &
                    (df_f['floors']<=f_floors) ]

    # DF for dataset summary: zipcode as index
    index = pd.Series(df_f['zipcode'].sort_values().unique(), name='zipcode')
    df1 = pd.DataFrame(index=index)
    # DF columns
    df1['count'] = df_f[['id', 'zipcode']].groupby('zipcode').count()
    df1['mdn_investment'] = df_f[['est_investment', 'zipcode']].groupby('zipcode').median()
    df1['mdn_price_zipcode'] = df_f[['price_mdn', 'zipcode']].groupby('zipcode').median()
    df1.reset_index(inplace=True)

    
    # DF for descriptive statistics: numeric features as index
    num_features = ['price', 'est_investment','est_costs',
                    'yr_built', 'condition', 'grade',
                    'floors', 'bedrooms', 'bathrooms',
                    'sqft_living', 'sqft_above','sqft_basement']

    descriptives = pd.DataFrame(index = num_features)
    # DF columns
    descriptives['min'] = df_f[num_features].min()
    descriptives['median'] = df_f[num_features].median()
    descriptives['max'] = df_f[num_features].max()
    descriptives['mean'] = df_f[num_features].mean()

    # Display
    st.header('Dataset overview')
    n_prop = df_f.shape[0]
    st.write('Number of properties selected:', n_prop)

    st.subheader('Table')
    st.dataframe( df_f.style.format({'price':'{:.2f}',
                                     'est_investment':'{:.2f}',
                                     'est_costs':'{:.2f}',
                                     'est_costs %':'{:.2f}',
                                     'price_mdn':'{:.2f}',
                                     'floors':'{:.1f}',
                                     'bathrooms':'{:.2f}'}))

    c1, c2 = st.columns(2)
    c1.subheader('Summary')
    c2.subheader('Descriptive Stastics')
    c1.dataframe(df1.style.format({'mdn_investment':"{:.2f}",
                                   'mdn_price_zipcode':'{:.2f}'}), height=460)
    
    c2.dataframe(descriptives.style.format("{:.0f}"), height=460)


    return df_f

def overview_plot( df ):
    # Displaying
    st.header('Price distribution overview')
    
    fig = plt.figure(figsize=(13, 6))
    specs = gridspec.GridSpec(nrows=1, ncols=1 , figure=fig)
    
    ax1 = fig.add_subplot(specs[0, 0])
    
    # Median prices of the selected properties
    df = df[['zipcode','price_mdn', 'price']].sort_values('price_mdn')#.reset_index()
    df['zipcode'] = df['zipcode'].astype(str)
    #df.columns=['zipcode', 'price']
    
    # Median prices of all properties 
    df2 = df[['zipcode', 'price_mdn', 'price']].sort_values('price_mdn')
    df2['zipcode'] = df2['zipcode'].astype(str)
    
    # Plot median
    sns.pointplot(data=df2, x='zipcode', 
                            y='price_mdn', 
                            markers = 's', 
                            linestyles='', ax=ax1)
    
    # Plot 
    sns.boxplot(data=df, y='price', x='zipcode', ax=ax1)
    
    ax1.set_title('Median price per zip codes (blue squares) and price distribution (buy recommendation)', fontsize=16)
    ax1.tick_params(axis='x', labelrotation=90)
    st.pyplot(fig, use_container_width=True)

    return None

def detailed_plots( df ):

    # Displaying
    # Displaying
    st.header('Estimated investment')
    c1, c2 = st.columns(2)
    c1.subheader('Total')
    df1 = df[['price','est_investment']].sum().reset_index()
    df1.columns=['x', 'US$']
    fig = px.bar(df1, x='x', y='US$',text_auto=True)
    c1.plotly_chart(fig, use_container_width=True)

    c2.subheader('Costs')
    fig = px.histogram(df, x='est_costs %')
    c2.plotly_chart(fig, use_container_width=True)

    st.header('Estimated investment per attribute')
    c1, c2 = st.columns(2)
    c1.subheader('Condition')
    fig = px.histogram(df, x='condition',y='est_investment', nbins=10)
    c1.plotly_chart(fig, use_container_width=True)

    c2.subheader('Grade')
    fig = px.histogram(df, x='grade',y='est_investment', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.subheader('Bedrooms')
    fig = px.histogram(df, x='bedrooms',y='est_investment', nbins=10)
    c1.plotly_chart(fig, use_container_width=True)

    c2.subheader('Bathrooms')
    fig = px.histogram(df, x='bathrooms',y='est_investment', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.subheader('Floors')
    fig = px.histogram(df, x='floors',y='est_investment', nbins=10)
    c1.plotly_chart(fig, use_container_width=True)

    c2.subheader('Basement')
    fig = px.histogram(df, x='basement',y='est_investment', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None


def portfolio_density( df, geofile ):
    st.header( 'Region overview' )

    # Region Price Map
    st.subheader( 'Price density' )

    df1 = df[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df1.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin( df1['ZIP'].tolist() )]

    region_price_map = folium.Map( location=[df['lat'].quantile(0.8), 
                                   df['long'].quantile(0.8) ],
                                   default_zoom_start=13 ) 

    folium.Choropleth( data = df1,
                       geo_data = geofile,
                       columns=['ZIP', 'PRICE'],
                       key_on='feature.properties.ZIP',
                       fill_color='YlOrRd',
                       fill_opacity = 0.7,
                       line_opacity = 0.2,
                       legend_name='AVG PRICE' ).add_to(region_price_map)

    folium_static( region_price_map )

    st.subheader( 'Portfolio density' )

    st.write( 'Zoom in and click on the property to see detailed information.' )

    # Base Map - Folium 
    density_map = folium.Map( location=[df['lat'].quantile(0.8), 
                            df['long'].quantile(0.8) ],
                            default_zoom_start=13 ) 

    marker_cluster = MarkerCluster().add_to( density_map )
    for name, row in df.iterrows():
        folium.Marker( [row['lat'], row['long'] ], 
            popup='Est. Invest.: US${0} Costs : {1}%. Features: {2} sqft, {3} bedrooms, {4} bathrooms, condition: {5}, grade: {6}, year built: {7}'.format( row['est_investment'],
                                        round(row['est_costs %'],2),
                                        row['sqft_living'],
                                        row['bedrooms'],
                                        row['bathrooms'],
                                        row['condition'],
                                        row['grade'],
                                        row['yr_built'] ) ).add_to( marker_cluster )

    folium_static( density_map )
    
    return None


# Here starts the program:
if __name__ == '__main__':
    # Set files
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
      
    # Get data
    data = get_data( path )
    geofile = get_geofile ( url )
    
    # Data transformation
    data = set_transformations( data )
    
    # Selecting 
    buy = select_properties( data, 0.30)
    
    # Overview data
    df_f = data_overview ( buy )
    
    # Overview plots
    overview_plot( df_f )
    
    # More detailed plots
    detailed_plots( df_f )
    
    # Ploting maps
    portfolio_density( df_f, geofile )