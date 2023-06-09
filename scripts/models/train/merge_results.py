import pandas as pd


forecast_horizon_hours = range(0,37)

test_or_train = 'test'


all_df = None
for forecast_horizon_hour in forecast_horizon_hours:
    print(forecast_horizon_hour)

    df = pd.read_csv(f'./results/errors_{forecast_horizon_hour}_{test_or_train}.csv', index_col=0)

    if forecast_horizon_hour>0:
        df = df[['prediction']]

    df = df.rename({'target': f'target_{forecast_horizon_hour}', 'prediction': f'+{forecast_horizon_hour}hours'}, axis=1)

    # if we want to results to be shift so that the index becomes the target time
    # otherwise the index is the init time
    # df = df.shift(forecast_horizon_hour*2, axis=0)

    if all_df is None:
        all_df = df
    else:
        all_df = all_df.join(df)


all_df.to_csv(f'./results/errors_all_{test_or_train}.csv')


