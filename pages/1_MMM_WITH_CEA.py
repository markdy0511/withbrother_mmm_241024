import pandas as pd
import streamlit as st
# Plotting the graph
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

import numpy as np

from scipy.optimize import linprog
import math

st.set_page_config(
    page_title="Report Assistant_MMM",
    page_icon="🥇",
    layout="wide",
)

import insert_logo 
insert_logo.add_logo("withbrother_logo.png")

# Find a font that supports Hangul (Korean)
font_path = "static/NanumGothic-Regular.ttf"
fontprop = fm.FontProperties(fname=font_path)

# Applying the font
plt.rcParams['font.family'] = fontprop.get_name()
# The user hasn't provided the dataframe itself, only a screenshot of the headers. 
# I'll create a sample dataframe based on the visible column names to proceed with plotting.

def calculate_icer_with_mc(df, min_cost, effect_columns):
    icers = pd.DataFrame()
    icers['Incr Cost'] = df['누적비용'] - min_cost
    for effect_col in effect_columns:
        df[effect_col] = df[effect_col].round(0)
        min_effect = df.loc[df['누적비용'] == min_cost, effect_col].values[0]
        icers['Incr Eff' + effect_col.split("_")[-1]] = df[effect_col] - min_effect

        # 5. ICER 계산 (비용 차이를 효과 차이로 나눈 값)
        icers['ICER' + effect_col.split("_")[-1]] = icers['Incr Cost'] / icers['Incr Eff' + effect_col.split("_")[-1]]
    return icers

def calculate_ceac(icer_list, wtp):
    return [np.mean([((0 < icer) and (icer <= threshold)) for icer in icer_list]) for threshold in wtp]

# 특정 WTP에서 CEAC 값을 노멀라이즈하는 함수
def normalize_ceac_by_wtp(ceac_values_per_media):
    # Transpose the CEAC values so that each WTP has values across all media
    ceac_values_per_media = np.array(ceac_values_per_media).T
    
    # WTP별로 합을 구하고 노멀라이즈
    normalized_ceac = []
    for ceac_values_at_wtp in ceac_values_per_media:
        summed_value = np.sum(ceac_values_at_wtp)
        if summed_value > 0:
            normalized_ceac.append(ceac_values_at_wtp / summed_value)
        else:
            normalized_ceac.append(np.zeros_like(ceac_values_at_wtp))
    
    return np.array(normalized_ceac).T  # 다시 Transpose하여 원래 형태로 반환

#보고서 유형 저장
if 'ider_df' not in st.session_state:
    st.session_state.icer_df = None

#보고서 유형 저장
if 'org_df' not in st.session_state:
    st.session_state.org_df = None

#기간 저장
if 'ceac' not in st.session_state:
    st.session_state.ceac = None


st.title('광고 예산 효율 및 추천(MMM)')


# 데이터 입력기
with st.sidebar: #원하는 소스를 만드는 곳
    st.sidebar.header('이곳에 데이터를 업로드하세요.')
    
    data = st.file_uploader(
        "매체 데이터 업로드 (Excel or CSV)",
        type=['xls','xlsx', 'csv'],
        key="uploader1"
    )

if data:
    st.header("1. ICER 분석")
    st.write("최저 비용의 매체를 기준으로 비용 효과를 분석합니다. (ICER : Incremental Cost Effectiveness Ratio)")
    with st.spinner("분석 중"):

        if st.session_state.icer_df is None:
            data1 = pd.read_excel('노스페이스_데모_데이터_v3.xlsx')
            df = pd.DataFrame(data1)

            df['일자'] = pd.to_datetime(df['일자'])

            filtered_df = df[df['일자'] == pd.to_datetime('2024-07-31')]
                # 2. 총비용을 기준으로 오름차순 정렬
            df_sorted = filtered_df.sort_values('누적비용')

            # 3. 총비용이 가장 낮은 값을 기준으로 차이를 계산
            min_cost = df_sorted['누적비용'].min()
            df_sorted['Incr Cost'] = df_sorted['누적비용'] - min_cost

            # 4. 효과 차이 계산
            df_sorted['효과'] = df_sorted['효과'].round(0)
            min_effect = df_sorted.loc[df_sorted['누적비용'] == min_cost, '효과'].values[0]
            df_sorted['Incr Eff'] = df_sorted['효과'] - min_effect

            # 5. ICER 계산 (비용 차이를 효과 차이로 나눈 값)
            df_sorted['ICER'] = (df_sorted['Incr Cost'] / df_sorted['Incr Eff']).round(1)
            df_sorted['매출'] = df_sorted['누적구매액']

            # Calculate the proportion of each value relative to the total sales
            df_sorted['매출 비중'] = ((df_sorted['누적구매액']/df_sorted['누적구매액'].sum())*100).round(2)

            df_final = df_sorted[['매체', '누적비용', 'Incr Cost', '효과', 'Incr Eff', 'ICER', '매출', '매출 비중']]
            st.session_state.icer_df = df_final
            st.session_state.org_df = df
            st.write(df_final)
        else:
            df_final = st.session_state.icer_df
            df = st.session_state.org_df
            st.write(st.session_state.icer_df)
    
    with st.expander("See Terminologies"):
        st.write(
            '''
            COST : 해당 기간에 투입한 비용 \n
            Incr cost : 가장 비용이 적게 투입한 광고 매체의 비용과의 비용 차이 \n
            Eff : 광고 노출이 1되었을 때, 얻을 수 있는 금액 효율 \n
            Incr Eff : 가장 비용이 적게 투입한 광고 매체의 효과와의 효과 차이 \n
            ICER : 효과 1을 올리기 위한 필요 비용 (음수일 경우, 상대적 효과가 없다고 해석)
            '''
        )

    st.write(
        '''
        - 틱톡에 가장 비용을 적게 사용하였습니다.
        - 틱톡과 대비하여 구글 검색광고, 메타가 유의미한 효과를 보이고 있습니다.
        - 구글 검색광고는 틱톡에 비해 1의 효과를 높이기 위해 9.1 의 비용이 더 필요합니다.
        - 메타는 6.1의 비용이 더 필요합니다.
        - 메타가 비효율적인 것이 아니냐라고 반문할 수 있으나, 메타가 매출의 굉장히 많은 비율(55.53%)을 차지하고 있기 때문에 비효율적이지만 안정적인 매출을 선택할 것인지에 대한 의사결정이 필요합니다.
        - 틱톡에 대비하여 구글검색광고는 효과 1원 올리는데9.1원 더 들고, 메타는 틱톡보다 효과 1원 올리는게 6.1원 더 듭니다. 그렇기에 틱톡이 구글이나 메타보다 효율이 떨어지는 상태는 아닙니다.
        '''
    )

    st.header("2. 비용-효과 추세 분석 그래프")

    st.write('비용 증가에 따른 효율 추세를 분석합니다.')

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="누적비용", y="효과", hue="매체", marker="o", dashes=False)

    plt.title("Effectiveness by Total Cost and Media")
    plt.xlabel("Total Cost (누적비용)")
    plt.ylabel("Effectiveness (효과)")
    plt.grid(True)
    # Display plot in Streamlit
    st.pyplot(plt)

    st.write(
        '''
        - 틱톡은 성과 자체가 비교적 크지는 않으나, 비용 대비 꾸준한 효과를 얻을 수 있을 것으로 판단됩니다. (선형의 변화가 유지됨)
        - 틱톡의 예산을 좀더 추가해보는 것을 고려할 수 있습니다.
        - 구글 검색 광고는 2백만원까지 효율이 점진적으로 증가하다, 이후로 효율이 급격히 떨어지는 모습입니다.
        - 예산 집행 중지 시점을 파악할 수 있습니다.
        - 메타는 점차 효율 상승의 포화가 이루지는 양상입니다.
        - 저비용-고효율 전략을 수립할 수 있습니다.
        '''
    )


    st.header("3. 확률론적 민감도분석 및 비용-효과 승인 커브 분석")
    st.write('효과 1을 얻기 위한 비용 지불의사별 매체 믹스 효율성 확률을 분석합니다.')

    if st.session_state.ceac is None:
        # Load the probability data
        df_prob = pd.read_excel("노스페이스_매체별_확률_필터링_v2.xlsx")

        # Step 1: Simulate random probabilities for pClick and pPurchase for 10,000 times for each media

        n_simulations = 100

        random_effects = pd.DataFrame()

        for i, row in df_prob.iterrows():
            # Generate random probabilities for pClick and pPurchase
            random_pClick = np.random.normal(row['pClick_mean'], row['pClick_std'], n_simulations)
            random_pPurchase = np.random.normal(row['pPurchase_mean'], row['pPurchase_std'], n_simulations)
            
            # Calculate the effect by multiplying the random probabilities with the '누적구매액'
            effect = random_pClick * random_pPurchase * row['누적구매액']
            
            # Store the effect values in the random_effects DataFrame
            random_effects[f'effect_{row["매체"]}'] = effect

        
        effect_pivot_df = random_effects.T
        # Renaming columns to "효과_0", "효과_1", etc.
        effect_pivot_df.columns = [f'효과_{i}' for i in range(effect_pivot_df.shape[1])]
        effect_pivot_df.index = effect_pivot_df.index.str.replace('effect_', '')

        # Step 4: Merge the pivoted effect data with media_df based on the media names
        final_merged_df = pd.merge(df_prob, effect_pivot_df, left_on='매체', right_index=True)
        # Step 2: Calculate ICER by comparing the costs and effects
        df_sorted_prob = final_merged_df.sort_values('누적비용')
        min_cost = df_sorted_prob['누적비용'].min()

        #st.write(df_sorted_prob)

        effect_columns = [col for col in df_sorted_prob.columns if '효과' in col]

        # Calculating ICER using the minimum cost reference
        icer_df = calculate_icer_with_mc(df_sorted_prob, min_cost, effect_columns)

        # Adding an index column for readability
        icer_df.index = df_sorted_prob['매체']

        #st.write(icer_df)

        # Define WTP thresholds (these can be adjusted)
        wtp_thresholds = np.arange(0, 100, 1)
        # Extracting only the ICER columns (which typically have names starting with 'ICER')
        icer_columns = [col for col in icer_df.columns if 'ICER' in col]

        # Create a dictionary of media and their corresponding ICER values
        icer_values = {media_name: row[icer_columns].values for media_name, row in icer_df.iterrows()}
        icer_values["틱톡"] = np.ones_like(icer_values["메타"])

        
        # Create CEAC plot
        plt.figure(figsize=(10, 6))

        # 각 미디어별로 WTP에 따른 CEAC 계산
        ceac_values_per_media = []
        for media, icer_list in icer_values.items():
            ceac_values_per_media.append(calculate_ceac(icer_list, wtp_thresholds))

        # 각 WTP에 대한 CEAC 값을 노멀라이즈
        normalized_ceac_values_per_wtp = normalize_ceac_by_wtp(ceac_values_per_media)

        extracted_probabilities = {}
        wtp_thresholds_list = wtp_thresholds.tolist()
        wtp = 10
        # 미디어별로 노멀라이즈된 CEAC 그래프 그리기
        for i, media in enumerate(icer_values.keys()):
            wtp_index = wtp_thresholds_list.index(wtp)
            extracted_probabilities[media] = normalized_ceac_values_per_wtp[i][wtp_index]
            plt.plot(wtp_thresholds, normalized_ceac_values_per_wtp[i], label=f'{media} (Normalized CEAC)')


        # Customize plot with additional information
        plt.title('CE Acceptability Curve')
        plt.xlabel('Willingness-to-Pay')
        plt.ylabel('% Iterations Cost-Effective')
        plt.legend(title="Strategy", loc='best', bbox_to_anchor=(1.05, 1))
        plt.grid(True)

        plt.tight_layout()
        st.pyplot(plt)
        
        st.session_state.ceac = {}
        st.session_state.ceac['icer_values'] = icer_values
        st.session_state.ceac['wtp_thresholds_list'] = wtp_thresholds_list
        st.session_state.ceac['extracted_probabilities'] = extracted_probabilities
        st.session_state.ceac['wtp_thresholds'] = wtp_thresholds
        st.session_state.ceac['normalized_ceac_values_per_wtp'] = normalized_ceac_values_per_wtp

    else:
        extracted_probabilities =  st.session_state.ceac['extracted_probabilities']
        wtp = 10
        # 미디어별로 노멀라이즈된 CEAC 그래프 그리기
        for i, media in enumerate(st.session_state.ceac['icer_values'].keys()):
            wtp_index = st.session_state.ceac['wtp_thresholds_list'].index(wtp)
            plt.plot(st.session_state.ceac['wtp_thresholds'], st.session_state.ceac['normalized_ceac_values_per_wtp'][i], label=f'{media} (Normalized CEAC)')


        # Customize plot with additional information
        plt.title('CE Acceptability Curve')
        plt.xlabel('Willingness-to-Pay')
        plt.ylabel('% Iterations Cost-Effective')
        plt.legend(title="Strategy", loc='best', bbox_to_anchor=(1.05, 1))
        plt.grid(True)

        plt.tight_layout()
        st.pyplot(plt)

    st.write(
        '''
        - Willingness-To-Pay(WTP) 는 ICER 값으로 해석합니다. 광고효율 1을 올리기 위한 지불 비용입니다.
        - WTP에 따른 매체별 믹스를 확인할 수 있습니다.
        '''
    )
    
    st.header("4. 예산 믹스 추천")

    # 입력 데이터: 채널별 데이터 (예시)
    df_rec = df_final

    df_probabilities = pd.DataFrame.from_dict(extracted_probabilities, orient='index')
    # Rename the columns to reflect the WTP values
    df_probabilities.columns = ['CEAC']

    # Reset the index so that media names become a column
    df_probabilities.reset_index(inplace=True)
    df_probabilities.rename(columns={'index': '매체'}, inplace=True)

    df_merged = pd.merge(df_rec, df_probabilities, on='매체', how='left')
    df_merged['CEAC'] = df_merged['CEAC'].replace(0, 1e-2).fillna(1e-2)

    #st.write(df_merged)

    # 총 예산 설정
    total_budget = st.text_input("예산을 입력해주세요.",41000000)
    if total_budget:
        total_budget = int(total_budget)
        # 목표 함수: 매출을 기반으로 예산 할당 최적화
        # 예산 할당 비율을 최적화하기 위해 ICER와 CEAC를 반영
        df_merged['revenue_per_budget'] = df_merged['매출'] / df_merged['누적비용']

        # 목적 함수: 최대화할 값 (마이너스를 취해 최소화 문제로 변환)
        c = (-1*df_merged['revenue_per_budget'] * df_merged['CEAC'] * df_merged["매출 비중"]).to_list()
        #st.write(df_merged['CEAC'], df_merged['revenue_per_budget'], c)

        # 제약 조건: 예산의 합이 total_budget을 넘지 않게
        A = [1] * len(df_merged)  # 각 채널의 예산 비율 합이 총 예산과 같아야 함
        b = [total_budget]
        
        if total_budget <= 53300000:
            upper_cap = 1.3
        else:
            upper_cap = math.ceil(total_budget*10/41000000) / 10
        # 경계 조건: 각 채널의 예산은 0 이상이어야 하고, 현재 예산 이상으로는 증가하지 않음
        bounds = [(0, budget*upper_cap) for budget in df_merged['누적비용']]
        
        #st.write(c,A,b,bounds)

        # 선형 계획법으로 최적화 실행
        result = linprog(c, A_eq=[A], b_eq=b, bounds=bounds, method='highs')

        # 최적화된 예산 할당 결과
        df_merged['추천 예산'] = (result.x).round(0)
        df_merged['기대 매출'] = (df_merged['추천 예산'] * df_merged['revenue_per_budget']).round(0)
        df_merged.loc['합계'] = df_merged.sum()

        # 결과 출력
        st.write("최적화된 예산 할당 결과를 출력합니다.")
        st.write(df_merged[['매체', '누적비용','매출', '추천 예산', '기대 매출']])
