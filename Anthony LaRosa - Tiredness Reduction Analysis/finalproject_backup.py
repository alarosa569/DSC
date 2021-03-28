# Anthony LaRosa
# 08/04/20
# DSC530-T301
# Professor Metzger

import numpy
import sys
import pandas
import nsfg
import thinkstats2
import seaborn
import matplotlib
import thinkplot
import statistics
from scipy import stats
from thinkstats2 import Mean, MeanVar, Var, Std, Cov

sleep_df = pandas.read_csv("SleepStudyData.csv")
print(sleep_df.head(5))
print(sleep_df.dtypes)

# enough_hist = thinkstats2.Hist(sleep_df.Enough)
# thinkplot.Hist(enough_hist)
# thinkplot.show(xlabel='Participant Answer', ylabel='Number of Responses', title='Do you think that you get enough sleep?')
#
hours_hist = thinkstats2.Hist(sleep_df.Hours)
# thinkplot.Hist(hours_hist)
# thinkplot.show(xlabel='Hours Slept', ylabel='Number of Responses', title='On average, how many hours of sleep do you get on a weeknight?')
#
# phreach_hist = thinkstats2.Hist(sleep_df.PhoneReach)
# thinkplot.Hist(phreach_hist)
# thinkplot.show(xlabel='Participant Answer', ylabel='Number of Responses', title='Do you sleep with your phone within arms reach?')
#
# phtime_hist = thinkstats2.Hist(sleep_df.PhoneTime)
# thinkplot.Hist(phtime_hist)
# thinkplot.show(xlabel='Participant Answer', ylabel='Number of Responses', title='Do you use your phone within 30 minutes of falling asleep?')
#
tired_hist = thinkstats2.Hist(sleep_df.Tired)
# thinkplot.Hist(tired_hist)
# thinkplot.show(xlabel='How tired are you?', ylabel='Number of Responses', title='Scale from 1 to 5, how tired are you throughout the day? (1 not tired, 5 being very tired)')
#
# breakfast_hist = thinkstats2.Hist(sleep_df.Breakfast)
# thinkplot.Hist(breakfast_hist)
# thinkplot.show(xlabel='Participant Answer', ylabel='Number of Responses', title='Do you typically eat breakfast?')

# print("Mean of Hours Slept " + str(numpy.mean(hours_hist)))
# print("Median of Hours Slept " + str(numpy.median(hours_hist)))
# print("Mode of Hours Slept " + str(stats.mode(sleep_df.Hours)))
# print("The Spread of Hours Slept is " + str(statistics.stdev(hours_hist)))
# print("The tails are ")
#
# print("Mean of Tired Scale " + str(numpy.mean(tired_hist)))
# print("Median of Tired Scale " + str(numpy.median(tired_hist)))
# print("Mode of Tired Scale " + str(stats.mode(sleep_df.Tired)))
# print("The Spread of Tired Scale is " + str(statistics.stdev(tired_hist)))
# print("The tails are ")

# PMF
# n_hours = hours_hist.Total()
# hours_pmf = hours_hist.Copy()
# for x, freq in hours_hist.Items():
#     hours_pmf[x] = freq / n_hours
# thinkplot.Hist(hours_pmf)
# thinkplot.show(xlabel='Number of Hours', ylabel='Probability', title='PMF of Hours Slept')
#
# n_tired = tired_hist.Total()
# tired_pmf = tired_hist.Copy()
# for x, freq in tired_hist.Items():
#     tired_pmf[x] = freq / n_tired
# thinkplot.Hist(tired_pmf)
# thinkplot.show(xlabel='Tired Scale: 1 Least Tired, 5 Most Tired', ylabel='Probability', title='PMF of Tiredness')

# CDF
# hours_cdf = thinkstats2.Cdf(sleep_df.Hours)
# thinkplot.Cdf(hours_cdf)
# thinkplot.show(xlabel='yee', ylabel='Probability', title='yeee')

# Plot 1 analytical distribution
# analytic_hours = sleep_df.Hours.dropna()
# mu, var = thinkstats2.TrimmedMeanVar(analytic_hours, p=0.01)
# print('Mean, Var', mu, var)
# sigma = numpy.sqrt(var)
# print('Sigma', sigma)
# xs, ps = thinkstats2.RenderNormalCdf(mu, sigma, low=0, high=12.5)
# thinkplot.Plot(xs, ps, label='model', color='0.6')
# analytic_hours_cdf = thinkstats2.Cdf(analytic_hours, label='data')
# thinkplot.PrePlot(1)
# thinkplot.Cdf(analytic_hours_cdf)
# thinkplot.show(title='Hours Slept Normal Analytical Model ', xlabel='Number of Hours', ylabel='CDF')

# #scatterplot
# thinkplot.Scatter(sleep_df.PhoneReach, sleep_df.Tired)
# thinkplot.Show(xlabel='Phone within Arm Length', ylabel='Tired Level', title='Phone Proximity vs Tiredness')
# thinkplot.Scatter(sleep_df.Hours, sleep_df.Tired)
# thinkplot.Show(xlabel='Hours Slept', ylabel='Tired Level', title='Hours Slept vs Tiredness')
#
# #covariance
# print("The Covariance is as follows:")
# print(numpy.cov(sleep_df.Hours, sleep_df.Tired))
#
# #pearsoncorr
# print("The Pearson is as follows:")
# print(sleep_df.corr(method='pearson'))
#
# #spearman
# print("The Spearman is as follows:")
# print(sleep_df.corr(method='spearman'))

#hypothesis test
# class CorrelationPermute(thinkstats2.HypothesisTest):
#
#     def TestStatistic(self, data):
#         xs, ys = data
#         test_stat = abs(thinkstats2.Corr(xs, ys))
#         return test_stat
#
#     def RunModel(self):
#         xs, ys = self.data
#         xs = numpy.random.permutation(xs)
#         return xs, ys
#
# hype_cleaned = sleep_df.dropna(subset=['Hours', 'Tired'])
# hype_data = hype_cleaned.Hours.values, hype_cleaned.Tired.values
# hype_ht = CorrelationPermute(hype_data)
# hype_pvalue = hype_ht.PValue()
# print(hype_pvalue)
# print(hype_ht.actual)
# print(hype_ht.MaxTestStat())
# the above is supporting the null hypothesis
# the opposite of my hypothesis

# least squares regression
def LeastSquares(xs, ys):
    meanx, varx = MeanVar(xs)
    meany = Mean(ys)

    slope = Cov(xs, ys, meanx, meany) / varx
    inter = meany - slope * meanx

    return inter, slope


sleep_regress = sleep_df.dropna(subset=['Hours', 'Tired'])
re_hours = sleep_regress.Hours
re_tired = sleep_regress.Tired
inter, slope = LeastSquares(re_hours, re_tired)
print(inter, slope)


def FitLine(xs, inter, slope):
    fit_xs = numpy.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys


fit_xs, fit_ys = FitLine(re_hours, inter, slope)

thinkplot.Scatter(re_hours, re_tired, color='blue', alpha=0.1, s=10)
thinkplot.Plot(fit_xs, fit_ys, color='white', linewidth=3)
thinkplot.Plot(fit_xs, fit_ys, color='red', linewidth=2)
thinkplot.show(xlabel="Hours", ylabel='Tiredness')
