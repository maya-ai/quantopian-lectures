#!/usr/bin/env python
# coding: utf-8

# # Introduction to pandas
# by Maxwell Margenot
# 
# Part of the Quantopian Lecture Series:
# 
# * [www.quantopian.com/lectures](https://www.quantopian.com/lectures)
# * [github.com/quantopian/research_public](https://github.com/quantopian/research_public)
# 
# 

# pandas is a Python library that provides a collection of powerful data structures to better help you manage data. In this lecture, we will cover how to use the `Series` and `DataFrame` objects to handle data. These objects have a strong integration with NumPy, covered elsewhere in the [lecture series](http://www.quantopian.com/lectures#Introduction-to-NumPy), allowing us to easily do the necessary statistical and mathematical calculations that we need for finance.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# With pandas, it is easy to store, visualize, and perform calculations on your data. With only a few lines of code we can modify our data and present it in an easily-understandable way. Here we simulate some returns in NumPy, put them into a pandas `DataFrame`, and perform calculations to turn them into prices and plot them, all only using a few lines of code.

# In[2]:


returns = pd.DataFrame(np.random.normal(1.0, 0.03, (100, 10)))
prices = returns.cumprod()
prices.plot()
plt.title('Randomly-generated Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc=0);


# So let's have a look at how we actually build up to this point!

# ## pandas Data Structures
# 
# ### `Series`
# 
# A pandas `Series` is a 1-dimensional array with labels that can contain any data type. We primarily use them for handling time series data. Creating a `Series` is as easy as calling `pandas.Series()` on a Python list or NumPy array.

# In[3]:


s = pd.Series([1, 2, np.nan, 4, 5])
print s


# Every `Series` has a name. We can give the series a name as a parameter or we can define it afterwards by directly accessing the name attribute. In this case, we have given our time series no name so the attribute should be empty.

# In[4]:


print s.name


# This name can be directly modified with no repercussions.

# In[5]:


s.name = "Toy Series"
print s.name


# We call the collected axis labels of a `Series` its index. An index can either passed to a `Series` as a parameter or added later, similarly to its name. In the absence of an index, a `Series` will simply contain an index composed of integers, starting at $0$, as in the case of our "Toy Series".

# In[6]:


print s.index


# pandas has a built-in function specifically for creating date indices, `date_range()`. We use the function here to create a new index for `s`.

# In[7]:


new_index = pd.date_range("2016-01-01", periods=len(s), freq="D")
print new_index


# An index must be exactly the same length as the `Series` itself. Each index must match one-to-one with each element of the `Series`. Once this is satisfied, we can directly modify the `Series` index, as with the name, to use our new and more informative index (relatively speaking).

# In[8]:


s.index = new_index
print s.index


# The index of the `Series` is crucial for handling time series, which we will get into a little later.

# #### Accessing `Series` Elements
# 
# `Series` are typically accessed using the `iloc[]` and `loc[]` methods. We use `iloc[]` to access elements by integer index and we use `loc[]` to access the index of the Series.

# In[9]:


print "First element of the series: ", s.iloc[0]
print "Last element of the series: ", s.iloc[len(s)-1]


# We can slice a `Series` similarly to our favorite collections, Python lists and NumPy arrays. We use the colon operator to indicate the slice.

# In[10]:


s.iloc[:2]


# When creating a slice, we have the options of specifying a beginning, an end, and a step. The slice will begin at the start index, and take steps of size `step` until it passes the end index, not including the end.

# In[11]:


start = 0
end = len(s) - 1
step = 1

s.iloc[start:end:step]


# We can even reverse a `Series` by specifying a negative step size. Similarly, we can index the start and end with a negative integer value.

# In[12]:


s.iloc[::-1]


# This returns a slice of the series that starts from the second to last element and ends at the third to last element (because the fourth to last is not included, taking steps of size $1$).

# In[13]:


s.iloc[-2:-4:-1]


# We can also access a series by using the values of its index. Since we indexed `s` with a collection of dates (`Timestamp` objects) we can look at the value contained in `s` for a particular date.

# In[14]:


s.loc['2016-01-01']


# Or even for a range of dates!

# In[15]:


s.loc['2016-01-02':'2016-01-04']


# With `Series`, we *can* just use the brackets (`[]`) to access elements, but this is not best practice. The brackets are ambiguous because they can be used to access `Series` (and `DataFrames`) using both index and integer values and the results will change based on context (especially with `DataFrames`).

# #### Boolean Indexing
# 
# In addition to the above-mentioned access methods, you can filter `Series` using boolean arrays. `Series` are compatible with your standard comparators. Once compared with whatever condition you like, you get back yet another `Series`, this time filled with boolean values.

# In[16]:


print s < 3


# We can pass *this* `Series` back into the original `Series` to filter out only the elements for which our condition is `True`.

# In[17]:


print s.loc[s < 3]


# If we so desire, we can group multiple conditions together using the logical operators `&`, `|`, and `~` (and, or, and not, respectively).

# In[18]:


print s.loc[(s < 3) & (s > 1)]


# This is very convenient for getting only elements of a `Series` that fulfill specific criteria that we need. It gets even more convenient when we are handling `DataFrames`.

# #### Indexing and Time Series
# 
# Since we use `Series` for handling time series, it's worth covering a little bit of how we handle the time component. For our purposes we use pandas `Timestamp` objects. Let's pull a full time series, complete with all the appropriate labels, by using our `get_pricing()` method. All data pulled with `get_pricing()` or using our Pipeline API will be in either `Series` or `DataFrame` format. We can modify this index however we like.

# In[19]:


symbol = "CMG"
start = "2012-01-01"
end = "2016-01-01"
prices = get_pricing(symbol, start_date=start, end_date=end, fields="price")


# We can display the first few elements of our series by using the `head()` method and specifying the number of elements that we want. The analogous method for the last few elements is `tail()`.

# In[20]:


print "\n", type(prices)
prices.head(5) 


# As with our toy example, we can specify a name for our time series, if only to clarify the name the `get_pricing()` provides us.

# In[21]:


print 'Old name: ', prices.name
prices.name = symbol
print 'New name: ', prices.name


# Let's take a closer look at the `DatetimeIndex` of our `prices` time series.

# In[22]:


print prices.index


# Notice that this `DatetimeIndex` has a collection of associated information. In particular it has an associated frequency (`freq`) and an associated timezone (`tz`). The frequency indicates whether the data is daily vs monthly vs some other period while the timezone indicates what locale this index is relative to. We can modify all of this extra information!
# 
# If we resample our `Series`, we can adjust the frequency of our data. We currently have daily data (excluding weekends) because `get_pricing()` pulls only data from market days. Let's up-sample from this daily data to monthly data using the `resample()` method.

# In[23]:


monthly_prices = prices.resample('M')
monthly_prices.head(10)


# The `resample()` method defaults to using the mean of the lower level data to create the higher level data. We can specify how else we might want the up-sampling to be calculated by specifying the `how` parameter.

# In[24]:


monthly_prices_med = prices.resample('M', how='median')
monthly_prices_med.head(10)


# We can even specify how we want the calculation of the new period to be done. Here we create a `custom_resampler()` function that will return the first value of the period. In our specific case, this will return a `Series` where the monthly value is the first value of that month.

# In[25]:


def custom_resampler(array_like):
    """ Returns the first value of the period """
    return array_like[0]

first_of_month_prices = prices.resample('M', how=custom_resampler)
first_of_month_prices.head(10)


# We can also adjust the timezone of a `Series` to adapt the time of real-world data. In our case, our time series is already localized to UTC, but let's say that we want to adjust the time to be 'US/Eastern'. In this case we use the `tz_convert()` method, since the time is already localized.

# In[26]:


eastern_prices = prices.tz_convert('US/Eastern')
eastern_prices.head(10)


# In addition to the capacity for timezone and frequency management, each time series has a built-in `reindex()` method that we can use to realign the existing data according to a new set of index labels. If data does not exist for a particular label, the data will be filled with a placeholder value. This is typically `np.nan`, though we can provide a fill method.
# 
# The data that we `get_pricing()` only includes market days. But what if we want prices for every single calendar day? This will include holidays and weekends, times when you normally cannot trade equities.  First let's create a new `DatetimeIndex` that contains all that we want.

# In[27]:


calendar_dates = pd.date_range(start=start, end=end, freq='D', tz='UTC')
print calendar_dates


# Now let's use this new set of dates to reindex our time series. We tell the function that the fill method that we want is `ffill`. This denotes "forward fill". Any `NaN` values will be filled by the *last value* listed. So the price on the weekend or on a holiday will be listed as the price on the last market day that we know about.

# In[28]:


calendar_prices = prices.reindex(calendar_dates, method='ffill')
calendar_prices.head(15)


# You'll notice that we still have a couple of `NaN` values right at the beginning of our time series. This is because the first of January in 2012 was a Sunday and the second was a market holiday! Because these are the earliest data points and we don't have any information from before them, they cannot be forward-filled. We will take care of these `NaN` values in the next section, when we deal with missing data.

# #### Missing Data
# 
# Whenever we deal with real data, there is a very real possibility of encountering missing values. Real data is riddled with holes and pandas provides us with ways to handle them. Sometimes resampling or reindexing can create `NaN` values. Fortunately, pandas provides us with ways to handle them. We have two primary means of coping with missing data. The first of these is filling in the missing data with  `fillna()`. For example, say that we want to fill in the missing days with the mean price of all days.

# In[29]:


meanfilled_prices = calendar_prices.fillna(calendar_prices.mean())
meanfilled_prices.head(10)


# Using `fillna()` is fairly easy. It is just a matter of indicating the value that you want to fill the spaces with. Unfortunately, this particular case doesn't make a whole lot of sense, for reasons discussed in the [lecture on stationarity](https://www.quantopian.com/lectures/integration-cointegration-and-stationarity) in the Lecture series. We could fill them with with $0$, simply, but that's similarly uninformative.
# 
# Rather than filling in specific values, we can use the `method` parameter, similarly to how the `reindex()` method works. We could use "backward fill", where `NaN`s are filled with the *next* filled value (instead of forward fill's *last* filled value) like so:

# In[30]:


bfilled_prices = calendar_prices.fillna(method='bfill')
bfilled_prices.head(10)


# But again, this is a bad idea for the same reasons as the previous option. Both of these so-called solutions take into account *future data* that was not available at the time of the data points that we are trying to fill. In the case of using the mean or the median, these summary statistics are calculated by taking into account the entire time series. Backward filling is equivalent to saying that the price of a particular security today, right now, tomorrow's price. This also makes no sense. These two options are both examples of look-ahead bias, using data that would be unknown or unavailable at the desired time, and should be avoided.
# 
# Our next option is significantly more appealing. We could simply drop the missing data using the `dropna()` method. This is much better alternative than filling `NaN` values in with arbitrary numbers.

# In[31]:


dropped_prices = calendar_prices.dropna()
dropped_prices.head(10)


# Now our time series is cleaned for the calendar year, with all of our `NaN` values properly handled. It is time to talk about how to actually do time series analysis with pandas data structures.

# #### Time Series Analysis with pandas
# 
# Let's do some basic time series analysis on our original prices. Each pandas `Series` has a built-in plotting method.

# In[32]:


prices.plot();
# We still need to add the axis labels and title ourselves
plt.title(symbol + " Prices")
plt.ylabel("Price")
plt.xlabel("Date");


# As well as some built-in descriptive statistics. We can either calculate these individually or using the `describe()` method.

# In[33]:


print "Mean: ", prices.mean()
print "Standard deviation: ", prices.std()


# In[34]:


print "Summary Statistics"
print prices.describe()


# We can easily modify `Series` with scalars using our basic mathematical operators.

# In[35]:


modified_prices = prices * 2 - 10
modified_prices.head(5)


# And we can create linear combinations of `Series` themselves using the basic mathematical operators. pandas will group up matching indices and perform the calculations elementwise to produce a new `Series`. 

# In[36]:


noisy_prices = prices + 5 * pd.Series(np.random.normal(0, 5, len(prices)), index=prices.index) + 20
noisy_prices.head(5)


# If there are no matching indices, however, we may get an empty `Series` in return.

# In[37]:


empty_series = prices + pd.Series(np.random.normal(0, 1, len(prices)))
empty_series.head(5)


# Rather than looking at a time series itself, we may want to look at its first-order differences or percent change (in order to get additive or multiplicative returns, in our particular case). Both of these are built-in methods.

# In[38]:


add_returns = prices.diff()[1:]
mult_returns = prices.pct_change()[1:]


# In[39]:


plt.title("Multiplicative returns of " + symbol)
plt.xlabel("Date")
plt.ylabel("Percent Returns")
mult_returns.plot();


# pandas has convenient functions for calculating rolling means and standard deviations, as well!

# In[40]:


rolling_mean = pd.rolling_mean(prices, 30)
rolling_mean.name = "30-day rolling mean"


# In[41]:


prices.plot()
rolling_mean.plot()
plt.title(symbol + "Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend();


# In[42]:


rolling_std = pd.rolling_std(prices, 30)
rolling_std.name = "30-day rolling volatility"


# In[43]:


rolling_std.plot()
plt.title(rolling_std.name);
plt.xlabel("Date")
plt.ylabel("Standard Deviation");


# Many NumPy functions will work on `Series` the same way that they work on 1-dimensional NumPy arrays.

# In[44]:


print np.median(mult_returns)


# The majority of these functions, however, are already implemented directly as `Series` and `DataFrame` methods.

# In[45]:


print mult_returns.median()


# In every case, using the built-in pandas method will be better than using the NumPy function on a pandas data structure due to improvements in performance. Make sure to check out the `Series` [documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html) before resorting to other calculations of common functions.

# ### `DataFrames`
# 
# Many of the aspects of working with `Series` carry over into `DataFrames`. pandas `DataFrames` allow us to easily manage our data with their intuitive structure. 
# 
# Like `Series`, `DataFrames` can hold multiple types of data, but `DataFrames` are 2-dimensional objects, unlike `Series`. Each `DataFrame` has an index and a columns attribute, which we will cover more in-depth when we start actually playing with an object. The index attribute is like the index of a `Series`, though indices in pandas have some extra features that we will unfortunately not be able to cover here. If you are interested in this, check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/version/0.18.1/advanced.html) on advanced indexing. The columns attribute is what provides the second dimension of our `DataFrames`, allowing us to combine named columns (all `Series`), into a cohesive object with the index lined-up.
# 
# We can create a `DataFrame` by calling `pandas.DataFrame()` on a dictionary or NumPy `ndarray`. We can also concatenate a group of pandas `Series` into a `DataFrame` using `pandas.concat()`.

# In[46]:


dict_data = {
    'a' : [1, 2, 3, 4, 5],
    'b' : ['L', 'K', 'J', 'M', 'Z'],
    'c' : np.random.normal(0, 1, 5)
}
print dict_data


# Each `DataFrame` has a few key attributes that we need to keep in mind. The first of these is the index attribute. We can easily include an index of `Timestamp` objects like we did with `Series`.

# In[47]:


frame_data = pd.DataFrame(dict_data, index=pd.date_range('2016-01-01', periods=5))
print frame_data


# As mentioned above, we can combine `Series` into `DataFrames`. Concatatenating `Series` like this will match elements up based on their corresponding index. As the following `Series` do not have an index assigned, they each default to an integer index. 

# In[48]:


s_1 = pd.Series([2, 4, 6, 8, 10], name='Evens')
s_2 = pd.Series([1, 3, 5, 7, 9], name="Odds")
numbers = pd.concat([s_1, s_2], axis=1)
print numbers


# We will use `pandas.concat()` again later to combine multiple `DataFrame`s into one. 

# Each `DataFrame` also has a `columns` attribute. These can either be assigned when we call `pandas.DataFrame` or they can be modified directly like the index. Note that when we concatenated the two `Series` above, the column names were the names of those `Series`.

# In[49]:


print numbers.columns


# To modify the columns after object creation, we need only do the following:

# In[50]:


numbers.columns = ['Shmevens', 'Shmodds']
print numbers


# In the same vein, the index of a `DataFrame` can be changed after the fact.

# In[51]:


print numbers.index


# In[52]:


numbers.index = pd.date_range("2016-01-01", periods=len(numbers))
print numbers


# Separate from the columns and index of a `DataFrame`, we can also directly access the values they contain by looking at the values attribute.

# In[53]:


numbers.values


# This returns a NumPy array.

# In[54]:


type(numbers.values)


# #### Accessing `DataFrame` elements
# 
# Again we see a lot of carryover from `Series` in how we access the elements of `DataFrames`. The key sticking point here is that everything has to take into account multiple dimensions now. The main way that this happens is through the access of the columns of a `DataFrame`, either individually or in groups. We can do this either by directly accessing the attributes or by using the methods we already are familiar with.

# In[55]:


symbol = ["CMG", "MCD", "SHAK", "WFM"]
start = "2012-01-01"
end = "2016-01-01"
prices = get_pricing(symbol, start_date=start, end_date=end, fields="price")
if isinstance(symbol, list):
    prices.columns = map(lambda x: x.symbol, prices.columns)
else:
    prices.name = symbol


# Here we directly access the `CMG` column. Note that this style of access will only work if your column name has no spaces or unfriendly characters in it.

# In[56]:


prices.CMG.head()


# We can also use `loc[]` to access an individual column like so.

# In[57]:


prices.loc[:, 'CMG'].head()


# Accessing an individual column will return a `Series`, regardless of how we get it.

# In[58]:


print type(prices.CMG)
print type(prices.loc[:, 'CMG'])


# Notice how we pass a tuple into the `loc[]` method? This is a key difference between accessing a `Series` and accessing a `DataFrame`, grounded in the fact that a `DataFrame` has multiple dimensions. When you pass a 2-dimensional tuple into a `DataFrame`, the first element of the tuple is applied to the rows and the second is applied to the columns. So, to break it down, the above line of code tells the `DataFrame` to return every single row of the column with label `'CMG'`. Lists of columns are also supported.

# In[59]:


prices.loc[:, ['CMG', 'MCD']].head()


# We can also simply access the `DataFrame` by index value using `loc[]`, as with `Series`.

# In[60]:


prices.loc['2015-12-15':'2015-12-22']


# This plays nicely with lists of columns, too.

# In[61]:


prices.loc['2015-12-15':'2015-12-22', ['CMG', 'MCD']]


# Using `iloc[]` also works similarly, allowing you to access parts of the `DataFrame` by integer index.

# In[62]:


prices.iloc[0:2, 1]


# In[63]:


# Access prices with integer index in
# [1, 3, 5, 7, 9, 11, 13, ..., 99]
# and in column 0 or 3
prices.iloc[[1, 3, 5] + range(7, 100, 2), [0, 3]].head(20)


# #### Boolean indexing
# 
# As with `Series`, sometimes we want to filter a `DataFrame` according to a set of criteria. We do this by indexing our `DataFrame` with boolean values.

# In[64]:


prices.loc[prices.MCD > prices.WFM].head()


# We can add multiple boolean conditions by using the logical operators `&`, `|`, and `~` (and, or, and not, respectively) again!

# In[65]:


prices.loc[(prices.MCD > prices.WFM) & ~prices.SHAK.isnull()].head()


# #### Adding, Removing Columns, Combining `DataFrames`/`Series`
# 
# It is all well and good when you already have a `DataFrame` filled with data, but it is also important to be able to add to the data that you have.
# 
# We add a new column simply by assigning data to a column that does not already exist. Here we use the `.loc[:, 'COL_NAME']` notation and store the output of `get_pricing()` (which returns a pandas `Series` if we only pass one security) there. This is the method that we would use to add a `Series` to an existing `DataFrame`.

# In[66]:


s_1 = get_pricing('TSLA', start_date=start, end_date=end, fields='price')
prices.loc[:, 'TSLA'] = s_1
prices.head(5)


# It is also just as easy to remove a column.

# In[67]:


prices = prices.drop('TSLA', axis=1)
prices.head(5)


# If we instead want to combine multiple `DataFrame`s into one, we use the `pandas.concat()` method.

# In[68]:


df_1 = get_pricing(['SPY', 'VXX'], start_date=start, end_date=end, fields='price')
df_2 = get_pricing(['MSFT', 'AAPL', 'GOOG'], start_date=start, end_date=end, fields='price')
df_3 = pd.concat([df_1, df_2], axis=1)
df_3.head()


# #### Missing data (again)
# Bringing real-life data into a `DataFrame` brings us the same problems that we had with it in a `Series`, only this time in more dimensions. We have access to the same methods as with `Series`, as demonstrated below.

# In[69]:


filled0_prices = prices.fillna(0)
filled0_prices.head(5)


# In[70]:


bfilled_prices = prices.fillna(method='bfill')
bfilled_prices.head(5)


# But again, the best choice in this case (since we are still using time series data, handling multiple time series at once) is still to simply drop the missing values.

# In[71]:


dropped_prices = prices.dropna()
dropped_prices.head(5)


# #### Time Series Analysis with pandas
# 
# Using the built-in statistics methods for `DataFrames`, we can perform calculations on multiple time series at once! The code to perform calculations on `DataFrames` here is almost exactly the same as the methods used for `Series` above, so don't worry about re-learning everything.
# 
# The `plot()` method makes another appearance here, this time with a built-in legend that corresponds to the names of the columns that you are plotting.

# In[72]:


prices.plot()
plt.title("Collected Stock Prices")
plt.ylabel("Price")
plt.xlabel("Date");


# The same statistical functions from our interactions with `Series` resurface here with the addition of the `axis` parameter. By specifying the `axis`, we tell pandas to calculate the desired function along either the rows (`axis=0`) or the columns (`axis=1`). We can easily calculate the mean of each columns like so:

# In[73]:


prices.mean(axis=0)


# As well as the standard deviation:

# In[74]:


prices.std(axis=0)


# Again, the `describe()` function will provide us with summary statistics of our data if we would rather have all of our typical statistics in a convenient visual instead of calculating them individually.

# In[75]:


prices.describe()


# We can scale and add scalars to our `DataFrame`, as you might suspect after dealing with `Series`. This again works element-wise.

# In[76]:


(2 * prices - 50).head(5)


# Here we use the `pct_change()` method to get a `DataFrame` of the multiplicative returns of the securities that we are looking at.

# In[77]:


mult_returns = prices.pct_change()[1:]
mult_returns.head()


# If we use our statistics methods to standardize the returns, a common procedure when examining data, then we can get a better idea of how they all move relative to each other on the same scale.

# In[78]:


norm_returns = (mult_returns - mult_returns.mean(axis=0))/mult_returns.std(axis=0)
norm_returns.loc['2014-01-01':'2015-01-01'].plot();


# This makes it easier to compare the motion of the different time series contained in our example.

# Rolling means and standard deviations also work with `DataFrames`.

# In[79]:


rolling_mean = pd.rolling_mean(prices, 30)
rolling_mean.columns = prices.columns


# In[80]:


rolling_mean.plot()
plt.title("Rolling Mean of Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend();


# For a complete list of all the methods that are built into `DataFrame`s, check out the [documentation](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html).

# # Next Steps
# 
# Managing data gets a lot easier when you deal with pandas, though this has been a very general introduction. There are many more tools within the package which you may discover while trying to get your data to do precisely what you want. If you would rather read more on the additional capabilities of pandas, check out the [documentation](http://pandas.pydata.org/pandas-docs/stable/).

# *This presentation is for informational purposes only and does not constitute an offer to sell, a solicitation to buy, or a recommendation for any security; nor does it constitute an offer to provide investment advisory or other services by Quantopian, Inc. ("Quantopian"). Nothing contained herein constitutes investment advice or offers any opinion with respect to the suitability of any security, and any views expressed herein should not be taken as advice to buy, sell, or hold any security or as an endorsement of any security or company.  In preparing the information contained herein, Quantopian, Inc. has not taken into account the investment needs, objectives, and financial circumstances of any particular investor. Any views expressed and data illustrated herein were prepared based upon information, believed to be reliable, available to Quantopian, Inc. at the time of publication. Quantopian makes no guarantees as to their accuracy or completeness. All information is subject to change and may quickly become unreliable for various reasons, including changes in market conditions or economic circumstances.*
