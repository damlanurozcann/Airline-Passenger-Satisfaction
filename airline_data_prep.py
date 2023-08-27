def airline_data_prep(dataframe):
    dataframe = dataframe.drop("id", axis=1)
    dataframe = dataframe.drop("Unnamed: 0", axis=1)  # We don't need those variables.

    dataframe.columns = [col.upper() for col in dataframe.columns]

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    cat_cols = [col for col in cat_cols if col not in ["SATISFACTION"]]  # Satisfaciton is our target value.

    dataframe = label_encoder(dataframe, "SATISFACTION")  ###  ! (1: satisfaction) & (0: neutral or dissatisfied)

    dataframe["ARRIVAL DELAY IN MINUTES"].fillna(dataframe["DEPARTURE DELAY IN MINUTES"], inplace=True)

    for col in num_cols:

        if check_outlier(dataframe, col):
            replace_with_thresholds(dataframe, col)

    dataframe.loc[(dataframe["AGE"] < 18), "NEW_AGE_CAT"] = "Child"
    dataframe.loc[(dataframe["AGE"] >= 18) & (dataframe["AGE"] < 35), "NEW_AGE_CAT"] = "mature"
    dataframe.loc[(dataframe["AGE"] >= 35) & (dataframe["AGE"] < 50), "NEW_AGE_CAT"] = "Senior"
    dataframe.loc[(dataframe["AGE"] >= 50), "NEW_AGE_CAT"] = "Old"

    # Industry Standards: short distance: x < 1000 km, middle distance: 1000 km < x < 3000 km, long distance x > 3000 km
    dataframe.loc[(dataframe["FLIGHT DISTANCE"] < 1000), "NEW_FLIGHT_DISTANCE_CAT"] = "Short_distance"
    dataframe.loc[(dataframe["FLIGHT DISTANCE"] >= 1000) & (dataframe["AGE"] < 3000), "NEW_FLIGHT_DISTANCE_CAT"] = "Middle_distance"
    dataframe.loc[(dataframe["FLIGHT DISTANCE"] >= 3000), "NEW_FLIGHT_DISTANCE_CAT"] = "Long_distance"

    dataframe["NEW_TOTAL_SERVICE_AVG"] = dataframe[["INFLIGHT WIFI SERVICE", "FOOD AND DRINK", "INFLIGHT ENTERTAINMENT", "INFLIGHT SERVICE", "ON-BOARD SERVICE"]].mean(axis=1)

    dataframe["NEW_TOTAL_COMFORT_AVG"] = dataframe[["SEAT COMFORT", "LEG ROOM SERVICE", "CLEANLINESS"]].mean(axis=1)

    dataframe["NEW_TOTAL_OUTSIDEPLANE_AVG"] = dataframe[["EASE OF ONLINE BOOKING", "GATE LOCATION", "ONLINE BOARDING", "BAGGAGE HANDLING", "CHECKIN SERVICE"]].mean(axis=1)

    dataframe["NEW_TOTAL_DELAY"] = dataframe[["DEPARTURE DELAY IN MINUTES", "ARRIVAL DELAY IN MINUTES"]].sum(axis=1)

    dataframe["NEW_GENDER_PLUS_TRAVEL"] = dataframe["GENDER"] + "-" + dataframe["TYPE OF TRAVEL"]

    # We created new variables so we need to use grap_col_ names again

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    binary_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # Update cat_cols

    cat_cols = [col for col in cat_cols if dataframe[col].dtypes == "O"]

    dataframe = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    Y = dataframe["SATISFACTION"]
    X = dataframe.drop(["SATISFACTION"], axis=1)

    print("Data Set is ready to use in any model. (X = All variables), (Y = SATISFACTION)")
    return X, Y