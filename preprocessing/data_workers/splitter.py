class Splitter:
    @staticmethod
    def name():
        return "basic"

    def split(self, dataframe, split_ratios):
        dataframe = dataframe.sample(frac=1, random_state=666)  # shuffle
        data_num = len(dataframe)
        train_split = int(split_ratios[0] / sum(split_ratios) * data_num)
        val_split = train_split + int(split_ratios[1] / sum(split_ratios) * data_num)
        return dataframe.iloc[:train_split], dataframe.iloc[train_split:val_split], dataframe.iloc[val_split:]
