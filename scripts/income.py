df["Date"] = pd.to_datetime(df["Date"])
df = df.groupby("Date")["gross income"].sum().reset_index()
