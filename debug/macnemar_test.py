import argparse
import pandas as pd
from scipy.stats import chi2


def main(args):
    #SeHGNN の予測ラベルのデータ成形
    SeHGNN_df = pd.read_table(filepath_or_buffer=args.SeHGNN)
    SeHGNN_df = SeHGNN_df.rename(columns={"0":"test_node_id","0.1":"node_type","2":"SeHGNN_pred_label","2.1":"true_label"})
    SeHGNN_df = SeHGNN_df.drop('node_type', axis=1)
    #提案手法の予測ラベルのデータ成形
    SeHGNN_ver2_df = pd.read_table(filepath_or_buffer=args.SeHGNN_ver2)
    SeHGNN_ver2_df = SeHGNN_ver2_df.rename(columns={"0":"test_node_id","0.1":"node_type","2":"SeHGNNver2_pred_label","2.1":"true_label"})
    SeHGNN_ver2_df = SeHGNN_ver2_df.drop('node_type', axis=1)
    #SeHGNN と 提案手法をマージ
    df = SeHGNN_df.merge(SeHGNN_ver2_df,on=["test_node_id","true_label"])

    #クロス集計表の作成
    # 11→ SeHGNN の予測が正しい and 提案手法の予測が正しい
    # 10→ SeHGNN の予測が正しい and 提案手法の予測が異なる
    # 01→ SeHGNN の予測が異なる and 提案手法の予測が正しい
    # 00→ SeHGNN の予測が異なる and 提案手法の予測が異なる
    df['macnemar'] = df.apply( lambda row: '11' if (row['SeHGNN_pred_label'] == row['true_label'] and row['SeHGNNver2_pred_label'] == row['true_label']) 
                              else '10' if (row['SeHGNN_pred_label'] == row['true_label'] and row['SeHGNNver2_pred_label'] != row['true_label']) 
                              else '01' if (row['SeHGNN_pred_label'] != row['true_label'] and row['SeHGNNver2_pred_label'] == row['true_label']) 
                              else '00', 
                              axis=1 )

    #クロス集計表の体格成分のみ抽出
    count_10 = df['macnemar'].value_counts()["10"] 
    count_01 = df['macnemar'].value_counts()["01"] 
    #カイ二乗値の計算
    chi_square_value = ((count_10 - count_01) **2) /(count_10+count_01)
    #マクネマー検定の実施
    p_value = chi2.sf(chi_square_value, 1)
    # 結果の出力（p値）
    print("p-value:", p_value) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ACM')
    parser.add_argument('--SeHGNN', type=str, default='None')
    parser.add_argument('--SeHGNN_ver2', type=str, default='None')
    args = parser.parse_args()
    main(args)











