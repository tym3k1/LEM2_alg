import pandas as pd
from typing import Dict, List, Callable, Tuple

class lem2Classifier:
    def __init__(self) -> None:
        self.rules = None
        self.decision_variables_name = None
        self.decision_variables = None
        self.df = None
    
    def split_dataframe_decisions_class(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        df_splitted = []
        for var in self.decision_variables:
            df_class = df[df[self.decision_variables_name] == var]
            df_splitted.append(df_class)
        return df_splitted
    
    def check_precision(self, k_col: str, mc: int) -> float:
        df = self.df
        suma = df.loc[df[k_col] == mc, self.decision_variables_name].value_counts()
        suma_sum = suma.sum()
        if len(suma) > 1:
            sum_jeden = suma[1]/suma_sum
            sum_zero = suma[0]/suma_sum
            return max(sum_jeden, sum_zero)
        return 1


    def most_common_value(self, df: pd.DataFrame, mc: int = 0, na: int = 0, k_col: int = 0, prec: float = 0) -> str and int:  #optymalizacja
        for col in df.iloc[:, :-1].columns:
            q_value = df[col].value_counts()
            most_common = q_value.idxmax()
            number_appearances = q_value.max()
            precision = self.check_precision(col, most_common)
            if na <= number_appearances and precision > prec:
                mc = most_common
                na = number_appearances
                k_col = col
                prec = precision
        return k_col, mc

    def check_inconsistency(self, df: pd.DataFrame, k_col: str, nw: int) -> bool:  #optymalizacja to df
        if len((df.loc[df[k_col] == nw, self.decision_variables_name] == 1).unique()) > 1:
            return True
        return False


    def one_rule(self, dk_ss: pd.DataFrame, dec_val: int) -> Dict and pd.DataFrame.index:  #optymalizacja
        rule = {}
        while not dk_ss.empty:
            if dk_ss.shape[1] > 1 and dk_ss.shape[0] > 0:
                k_col, mc = self.most_common_value(dk_ss)
                bool_val_of = self.check_inconsistency(self.df, k_col, mc)
                if not bool_val_of:
                    rule[k_col] = mc
                    rule[self.decision_variables_name] = dec_val
                    dk_ss = dk_ss[dk_ss[k_col] == mc]
                    index = dk_ss.index
                    return rule, index
                rule[k_col] = mc
                dk_ss = dk_ss[dk_ss[k_col] == mc]
                dk_ss = dk_ss.drop(columns=k_col)
                index = dk_ss.index
            else:
                print("The columns have too little diversity. Try to increase it using feature engineering.")
                rule[k_col] = mc
                rule[self.decision_variables_name] = None
                return rule, index

    
    def LEM2_algorithm(self, df: pd.DataFrame) -> List:
        self.df = df
        self.decision_variables = list(self.df.iloc[:, -1].unique())
        self.decision_variables_name = df.columns[-1]
        list_of_splitted_df = self.split_dataframe_decisions_class(df)
        self.rules = list()
        pdk_ = {}
        for item in list_of_splitted_df: #  from list of df take df with full coverage of 0/1
            dec_value =  int(item.iloc[:, -1].unique())
            while not item.empty: #  not hard to explain xD
                dk_ss = item.copy()
                rule, idx = self.one_rule(dk_ss, dec_value)
                self.out_of(rule, pdk_)
                self.rules.append(rule)
                item = item.drop(idx)
        print(pdk_)
        self.rules = [dict(s) for s in set(frozenset(d.items()) for d in self.rules)] #  try with and wthout
        return self.rules 

    
        # Metoda trenowania modelu
    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = pd.concat([X, y], axis=1)
        self.LEM2_algorithm(df)
        return self

    # Metoda do przewidywania na nowych danych
    def predict(self, X: pd.DataFrame) -> pd.Series:
        predictions = []
        for idx, row in X.iterrows():
            for rule in self.rules:
                rule_matches = True
                for feature, value in rule.items():
                    if feature != self.decision_variables_name and row[feature] != value:
                        rule_matches = False
                        break
                if rule_matches:
                    if rule.get(self.decision_variables_name, None) == None:
                        predictions.append(False)
                    else:
                        predictions.append(True)
                    break  # Zakończ sprawdzanie kolejnych reguł po znalezieniu pasującej
            else:  # Ta część zostanie wykonana, gdy nie znajdziemy pasującej reguły
                predictions.append(False)
        return predictions
    
    def out_of(self, dic,res):  
        for key, value in dic.items():
            if value is not None and key not in list(res.keys()):
                res[key] = 1
            res[key] += 1
        return res