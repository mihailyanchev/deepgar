import json
from datetime import datetime
import pandas as pd

start_period_common = "2000Q1"
end_period_common = "2021Q4"

def fetch_gdp_data(qoq_growth=True, start_period=start_period_common, end_period=end_period_common, period_q=1):

    gdp_df = pd.read_csv("preprocessed_data/gdp.csv", index_col=0)

    cond_1 = (gdp_df.index >= start_period)
    cond_2 = (gdp_df.index <= end_period)

    gdp_df = gdp_df[cond_1 & cond_2]
    #gdp_df = gdp_df[cond_1]



    if qoq_growth:
        print("Preprocessing GDP data")
        gdp_df = gdp_df.pct_change(periods=period_q).iloc[period_q:, ]

    return gdp_df


def fetch_snp_data(qoq_growth=True, start_period=start_period_common, end_period=end_period_common, period_q=1):

    df = pd.read_csv("preprocessed_data/snp500.csv", index_col=0)

    cond_1 = (df.index >= start_period)
    #cond_2 = (indprod_df.index <= end_period)

    #indprod_df = indprod_df[cond_1 & cond_2]
    df = df[cond_1]

    if qoq_growth:
        print("Preprocessing S&P 500 data")
        df = df.pct_change(periods=period_q).iloc[period_q:, ]

    df = df.shift(periods=-1)

    return df

def fetch_sentiment_data(qoq_growth=True, start_period=start_period_common, end_period=end_period_common, period_q=1):

    sent_df = pd.read_csv("preprocessed_data/sentiment.csv", index_col=0)

    cond_1 = (sent_df.index >= start_period)
    #cond_2 = (sent_df.index <= end_period)

    #sent_df = sent_df[cond_1 & cond_2]
    sent_df = sent_df[cond_1]

    if qoq_growth:
        print("Preprocessing sentiment data")
        sent_df = sent_df.pct_change(periods=period_q).iloc[period_q:, ]

    sent_df = sent_df.shift(periods=-1)

    return sent_df

def fetch_us_bond_data(qoq_growth=True, start_period=start_period_common, end_period=end_period_common, period_q=1):

    sent_df = pd.read_csv("preprocessed_data/us_bond_yield.csv", index_col=0)

    cond_1 = (sent_df.index >= start_period)
    #cond_2 = (sent_df.index <= end_period)

    #sent_df = sent_df[cond_1 & cond_2]
    sent_df = sent_df[cond_1]

    if qoq_growth:
        print("Preprocessing US bond yield data")
        sent_df = sent_df.pct_change(periods=period_q).iloc[period_q:, ]

    sent_df = sent_df.shift(periods=-1)

    return sent_df

def main():
    '''
    Downloading all data.
    Exporting in csv.
    Writing to master log.
    '''

    gdp_df_growths = fetch_gdp_data()
    snp_growths = fetch_snp_data()
    sentiment = fetch_sentiment_data()
    us_bond_yield = fetch_us_bond_data()

    for country in list(gdp_df_growths.columns):
        print("Preparing file for %s" % country)
        df_list = []
        df_list.append(gdp_df_growths[country])
        #df_list.append(ind_prod_growths[country])
        df_list.append(sentiment[country])
        #df_list.append(clifs_growths[country])
        df_list.append(snp_growths['SNP_VOL'])
        df_list.append(snp_growths['SNP_CLOSE'])
        #df_list.append(sse_growths['SSE_VOL'])
        #df_list.append(sse_growths['SSE_CLOSE'])
        #df_list.append(brent['BRENT'])
        #df_list.append(ea_pmi['EA_PMI'])
        df_list.append(us_bond_yield['US_BOND_YIELD'])

        final_df = (pd.concat(df_list, axis=1))
        final_df = final_df.iloc[1: , :]

        #print(final_df)
        final_df.set_axis(['GDP', 
            'SENTIMENT', 
            'SNP_VOL', 
            'SNP_CLOSE',  
            'US_BOND_YIELD'], axis=1, inplace=True)

        cond_2 = (final_df.index <= end_period_common)

        final_df = final_df[cond_2]

        #final_df.set_axis(['GDP', 'SENTIMENT'], axis=1, inplace=True)
        now = datetime.now()
        final_df.to_csv('data_'+country+'_'+now.strftime('%d.%m.%Y_%H.%M.%S')+'.csv')
        print("Exporting file to csv format at "+now.strftime('%d.%m.%Y_%H.%M.%S'))

    print("Complete!")

if __name__ == "__main__":
    main()