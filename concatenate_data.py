import pandas as pd

bariba = pd.read_excel("1_formative_research/data/[Bariba] Benin Questionnaire d'Enquete_WIDE.xlsx")
dendi = pd.read_excel("1_formative_research/data/[Dendi] Benin Questionnaire d'Enquete_WIDE.xlsx")
fon = pd.read_excel("1_formative_research/data/[Fon] Benin Questionnaire d'Enquete_WIDE.xlsx")


# add form_language column
bariba['form_language'] = 'bariba'
dendi['form_language'] = 'dendi'
fon['form_language'] = 'fon'

# create one master df
master = pd.concat([bariba, dendi, fon], ignore_index=True)

master.to_csv("1_formative_research/table_analysis/data/1_raw/concatenated_data.csv", index=False)
