import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import *

st.set_option('deprecation.showPyplotGlobalUse', False)

clients = pd.DataFrame()
loan = pd.DataFrame()
target = pd.DataFrame()
job = pd.DataFrame()
salary = pd.DataFrame()
last_credit = pd.DataFrame()
close_loan = pd.DataFrame()

dir_ = 'datasets_hw1'

dataframes = {}
for filename in os.listdir(dir_):
    df_name = filename.replace('D_', '').removesuffix('.csv')
    file_path = os.path.join(dir_, filename)
    df = pd.read_csv(file_path)
    globals()[df_name] = dataframes[df_name] = pd.read_csv(file_path)

for df_name, df in dataframes.items():
    df.columns = [col.lower() for col in df.columns]

st.markdown("""**Справочники**:
* `work` - трудоустройство (раб/не раб/неизв)
* `pens` - пенсионер

**Фактовые таблицы**:
* `target` (agreement) - отклики клиентов на компанию (client -> target)
* `last_credit` - последний займ (client -> credit + fst_payment + term)
* `close_loan` - статусы кредитов (loan -> is_closed)

**EM таблицы**:
* `salary` - доход (client -> income [personal + family])
* `job` - работа (client -> industry + title + department + work_time [months])
* `clients` - клиенты (client -> ...)

**Others**:
* `loan` - link (loan <-> client)""")
st.markdown("---")

st.markdown("""Нет смысла проверять справочниики, разве что за орфографией.

Что стоит проверить в остальных таблицах:
* Ограничения целостности
    * Дубликаты (записали два раза - например, читали из кафки). Как полные, так и по ключу.
    * FK - ссылки на измерения и справочники существующие.
* Пропущенные значения.
* Ошибки ввода данных, которые ведут к нарушению бизнес-логики, выбросам и другим неприятным последствиям, которые не позволят, например, сделать нам построить качественную модель (ап-лифт? вроде так называется).]

Сначала можем пройтись по каждой табличке в частности, а потом посмотреть на итоговую таблицу, в которую уже сведем все измерения.""")
st.markdown("## Data processing + EDA")

for df_name, df in dataframes.items():
    df.columns = [col.lower() for col in df.columns]

clients.id = clients.id.astype(str)
target.agreement_rk = target.agreement_rk.astype(str)
target.id_client = target.id_client.astype(str)
target.target = target.target.astype(int)
job.id_client = job.id_client.astype(str)
salary.id_client = salary.id_client.astype(str)
last_credit.id_client = last_credit.id_client.astype(str)
loan.id_client = loan.id_client.astype(str)
loan.id_loan = loan.id_loan.astype(str)
close_loan.id_loan = close_loan.id_loan.astype(str)

##########################################################################################
st.markdown('###### loan')
st.markdown('Размерность таблицы:')
st.write(loan.shape)
st.markdown('Семпл данных:')
st.write(loan.head())

st.markdown("""Надо запомнить, что долгов у нас 21к+. 
Посмотрим потом, сколько долгов будет в других измерениях + сколько клиентов (вероятно, клиенты брали не по одному долгу).

Проверим на дубли.""")

cmd = "find_duplicates(df=loan, key_cols=['id_loan', 'id_client'], return_bool=True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

st.markdown("""Проверим также на пропущенные значения.""")
cmd = "find_missing_values(df=loan, return_bool=True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

_ = (
    loan
    .groupby('id_client')
    .count()
    .sort_values('id_loan', ascending=False)
    .plot(kind='hist', title='Распределение количества кредитов на одного клиента банка')
    .figure
)
st.pyplot(_)
st.markdown('Как можно заметить, в основном клиенты банка берут по 1-4 кредита и существенно реже больше.')

##########################################################################################
st.markdown('---')
st.markdown('###### target')
st.markdown('Размерность таблицы:')
st.write(target.shape)
st.markdown('Семпл данных:')
st.write(target.head())

st.markdown("""Клиентов, попавших под коммуникацию банка, 15к+. Запомнили.

Проверим, как и в прошлый раз - дубли и пропущенные значения.""")

cmd = "find_duplicates(target, ['agreement_rk'], True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')
cmd = "find_duplicates(target, ['id_client', 'target'], True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

cmd = "find_missing_values(target, True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

st.markdown("Посмотрим на распределение откликов клиентов.")
_ = (
    target
    .groupby('target')
    .count()
    .drop('agreement_rk', axis=1)
    .rename({'id_client': 'was_affected'}, axis=1)
     / target.shape[0] * 100
).round(2)
st.write(_)

_ = (
    target
    .groupby('target')
    .count()
    .drop('agreement_rk', axis=1)
    .rename({'id_client': 'was_affected'}, axis=1)
    .reset_index()
    .plot(kind='pie', y='was_affected', title='Сколько клиентов банка отреагировали на коммуникацию.')
    .figure
)
st.pyplot(_)
st.markdown("Примерно 12% клиентов, которым были посланы коммуникации, откликнулись на предложение.")

##########################################################################################
st.markdown('---')
st.markdown('###### last_credit')
st.markdown('Размерность таблицы:')
st.write(last_credit.shape)
st.markdown('Семпл данных:')
st.write(last_credit.head())

cmd = """find_duplicates(last_credit, 
                ['credit', 'term', 'fst_payment', 'id_client'], 
                True)"""
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

cmd = "find_missing_values(last_credit, True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

st.markdown("""**NOTE**: Уже что в первой строчке вижу, что первоначальный взнос почему-то больше кредита...""")

tmp = last_credit[last_credit.fst_payment > last_credit.credit]
tmp['over'] = np.round(tmp.fst_payment.values / tmp.credit.values - 1, 2)
st.markdown("Семпл таких данных:")
st.write(tmp.sort_values('over', ascending=False).head())
st.write(f"Количество кредитов c первым платежом больше самого кредита: {tmp.shape[0]}")

tmp = last_credit[last_credit.fst_payment == last_credit.credit]
st.write(f"Количество кредитов c первым платежом равным кредиту: {tmp.shape[0]}")

st.markdown("""Ну такого точно не должно быть. Даже если это не взнос, а первый платеж. Кредиты часто закрываются раньше, но не первым же платежом. Ну и тем более платежи больше самого кредита. Не говоря уже о том, что если это именно взносы, то уж никакой банк не будет заключать кредиты со взносом >= сумме кредита.

Предлагаю оставить только те кредиты, где первый платеж/взнос меньше кредита.""")

last_credit = last_credit[last_credit.fst_payment < last_credit.credit]
st.write(f"Количество семплов теперь: {last_credit.shape[0]}")

##########################################################################################
st.markdown('---')
st.markdown('###### close_loan')
st.markdown('Размерность таблицы:')
st.write(close_loan.shape)
st.markdown('Семпл данных:')
st.write(close_loan.head())

cmd = "find_missing_values(close_loan, True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

cmd = "find_duplicates(close_loan, ['id_loan'], True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

st.markdown("Посмотрим на процент погашенных кредитов.")
_ = (
    close_loan
    .groupby('closed_fl')
    .count()
    .rename({'id_loan': 'was_closed'}, axis=1)
     / close_loan.shape[0] * 100
).round(3)
st.write(_)

_ = (
    close_loan
    .groupby('closed_fl')
    .count()
    .rename({'id_loan': 'was_closed'}, axis=1)
    .reset_index()
    .plot(kind='pie', y='was_closed', title='Количество закрытых кредитов на момент сбора данных')
    .figure
)
st.pyplot(_)
st.markdown("""Получается на данный момент около 54% долгов закрыты. Остальные, видимо, еще действующие или уже списаны.""")

##########################################################################################
st.markdown('---')
st.markdown('###### salary')
st.markdown('Размерность таблицы:')
st.write(salary.shape)
st.markdown('Семпл данных:')
st.write(salary.head())

cmd = "find_missing_values(salary, True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')
cmd = "find_duplicates(salary, ['id_client'], True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

st.markdown("""**NOTE**: Нашли дубликаты. Полные или только по ключу?""")
cmd = """(
    find_duplicates(salary, ['id_client'], False)
    .equals(
        find_duplicates(salary, 
                        ['id_client', 'personal_income', 'family_income'], 
                        False)
    )
)"""
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')
st.write(f"Количество дублей в таблице `salary`: {find_duplicates(salary, ['id_client'], False).shape[0]}.")
st.markdown("""300 дубликатов, причем полных. Можно списать на какой-нибудь забагованный ETL-процесс.
Думаю, надо определенно удалять - пользы не несут, при этом при джойне еще замножатся строки + плохо повлияет на модель.""")

salary = drop_duplicates(df=salary, key_cols=['id_client'])
st.write(f"Количество семплов теперь: {salary.shape[0]}.")

st.markdown("""Посмотрим теперь на распределение доходов у клиентов банка.""")
income_counts = salary.groupby('personal_income')['id_client'].count().reset_index()

# Create a histogram plot
plt.figure(figsize=(10, 6))
plt.hist(income_counts['personal_income'], bins=20, edgecolor='k', density=True)
plt.title('Распределение личного дохода у клиентов банка')
plt.xlabel('Доход')
plt.ylabel('Процент')
plt.grid(True)
st.pyplot()

st.markdown("""Видно, что в основном личный доход клиентов сконцентрирован на значениях 0-50к с уклоном в 15-20к. Хвост распределения попадает на диапазон 50-150к и также еще выбросы от 150к и выше.""")


family_income_counts = salary.groupby('family_income')['id_client'].count().reset_index()
family_income_counts.rename(columns={'id_client': 'Count'}, inplace=True)
family_income_percentages = family_income_counts['Count'] / salary.shape[0]
plt.figure(figsize=(8, 8))
plt.pie(family_income_percentages, labels=family_income_counts['family_income'], autopct='%1.1f%%', startangle=140)
plt.title('Распределение доходов семьи у клиентов банка.')
st.pyplot()

st.markdown("""Видно, что по доходам семей все получается примерно также - основная масса распределения лежит в диапазоне от 10 до 50к, на остальное приходится менее 15% распределения.""")
st.markdown("""**NOTE**: Посмотрим еще на ошибки. Есть ли клиенты, у которых личный доход больше дохода семьи?""")

_ = salary['family_income'].apply(lambda x: x.split(' ')[-2]).astype(int)
st.write(salary[salary.personal_income > _].head())
st.write(f"Количество клиентов с личным доходом больше семейного дохода: {salary[salary.personal_income > _].shape[0]}.")

st.markdown("""По моей логике доход члена семьи не может быть больше дохода всей семьи. Либо снова ошибка с загрузкой данных в хранилище, либо ошибки при заполнении клиентами персональной анкеты.

Энивей, если мы потом собираемся строить модель на таких данных, то лучше тоже дропнуть.""")

salary = salary[salary.personal_income <= _]
st.write(f"Количество семплов теперь: {salary.shape[0]}.")

##########################################################################################
st.markdown('---')
st.markdown('###### job')
st.markdown('Размерность таблицы:')
st.write(job.shape)
st.markdown('Семпл данных:')
st.write(salary.head())

cmd = "find_duplicates(tmp, ['id_client'], True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')
cmd = "find_missing_values(job, True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

st.markdown("""**NOTE**: Нашли пропущенные значения. Полностью NULL записи или какие-то отдельные значения?""")

st.code("""find_missing_values(job, False)""", language='python')
st.write(find_missing_values(job, False).head())
st.write(f"Количество семплов: {find_missing_values(job, False).shape[0]}.")

st.markdown("""Получается, по каким-то клиентам инфы нет совсем. Кажется, что не должно на что-либо повлиять. Можно дропнуть, можно оставить. В конце концов при джойне все равно эти колонки будут и значения будут null, поэтому смысла дропать не вижу.


Но для целей анализа эти данные уберем, чтобы не мешались.""")

tmp = job[~job.isna().any(axis=1)]
tmp.work_time = tmp.work_time.astype(int)

st.markdown("""Посмотрим еще на распределение стажа.""")
_ = (
    tmp.groupby('work_time')
    .count()
    .drop(['gen_title', 'job_dir', 'id_client'], axis=1)
    .rename(columns={'gen_industry': 'Frequency'})
    .reset_index()
)

# Create a histogram plot
plt.figure(figsize=(10, 6))
plt.hist(_['work_time'], bins=20, edgecolor='k', density=True)
plt.title('Распределение стажа на последнем рабочем месте.')
plt.xlabel('Стаж (мес.)')
plt.ylabel('Частота')
plt.grid(True)
st.pyplot()

st.markdown("""Получается достаточно странный график. Похоже на выбросы. Идем разбираться.""")

_ = (
    tmp
    .groupby('work_time')
    .count()
    .drop(['gen_title', 'job_dir', 'id_client'], axis=1)
    .rename({'gen_industry': 'freq'}, axis=1)
    .reset_index()
    .sort_values('work_time', ascending=False)
)
st.write(_)

st.markdown("""**NOTE**: В топе датафрейма видны какие-то явные ошибки, потому что 1500 месяцев - это 125 лет работы, не говоря про значения больше. Ограничим работу на последнем месте в 60 лет.""")
filtered_data = tmp[tmp['work_time'] < 60 * 12]

_ = (
    filtered_data.groupby('work_time')
    .count()
    .drop(['gen_title', 'job_dir', 'id_client'], axis=1)
    .rename(columns={'gen_industry': 'Frequency'})
    .reset_index()
)
plt.figure(figsize=(10, 6))
plt.hist(_['work_time'], weights=_['Frequency'], color='skyblue', bins=50)
plt.title('Распределение стажа на последнем рабочем месте (truncated 60 лет).')
plt.xlabel('Стаж (мес.)')
plt.ylabel('Частота')
plt.grid(True)
st.pyplot()

st.markdown("""Все равно присутствуют выбросы в районе 600 (оно и понятно). Можно попытаться сделать график более детализированным, ограничившись максимальным стажем в 30 лет на последнем месте (что, на самом деле, и более реалистично).""")

filtered_data = tmp[tmp['work_time'] < 30 * 12]
_ = (
    filtered_data.groupby('work_time')
    .count()
    .drop(['gen_title', 'job_dir', 'id_client'], axis=1)
    .rename(columns={'gen_industry': 'Frequency'})
    .reset_index()
)
plt.figure(figsize=(10, 6))
plt.hist(_['work_time'], weights=_['Frequency'], color='skyblue', bins=50)
plt.title('Распределение стажа на последнем рабочем месте (truncated 30 лет).')
plt.xlabel('Стаж (мес.)')
plt.ylabel('Частота')
plt.grid(True)
st.pyplot()

st.markdown("""Здесь (на следующем графике) можно заметить, что частоты, кратные 6 месяцам, представляют собой пики распределения. Вряд ли это связано с желанием уйти из компании ровно в 1.5 года или 3 года работы. Вероятно респонденты предпочитают округлять свой стаж до ближайшего целого полугода, чтобы не вдаваться в детали.""")

filtered_data = tmp[tmp['work_time'] < 5 * 12]
_ = (
    filtered_data.groupby('work_time')
    .count()
    .drop(['gen_title', 'job_dir', 'id_client'], axis=1)
    .rename(columns={'gen_industry': 'Frequency'})
    .reset_index()
)
plt.figure(figsize=(10, 6))
plt.hist(_['work_time'], weights=_['Frequency'], color='skyblue', bins=50)
plt.title('Распределение стажа на последнем рабочем месте (truncated 30 лет).')
plt.xlabel('Стаж (мес.)')
plt.ylabel('Частота')
plt.grid(True)
st.pyplot()

st.markdown("""Почистим датафрейм на более менее реалистичные цифры стажа. Ограничим его 60 годами.""")

job = job.loc[job.work_time <= 60 * 12, :]
st.write(f"Количество семплов теперь: {job.shape[0]}.")

st.markdown("Посмотрим на другие распределения.")
_ = (
    tmp
    .gen_title
    .value_counts()
).plot(kind='pie', title='Распределение по названию должности')
st.pyplot(_.figure)
st.markdown("""Немношк налезло, но в целом разобрать можно. В основном все специалисты/рабочие, видимо также из простоты выбора (потому что работников в сфере услуг в развивающейся/развитой экономике должно быть по идее побольше). Ну и понятно, что руководителей меньше по количеству, чем линейного персонала.""")


##########################################################################################
st.markdown('---')
st.markdown('###### clients')

st.markdown("""Напоминаю описание данных:

* ID — идентификатор записи;
* AGE — возраст клиента;
* GENDER — пол клиента (1 — мужчина, 0 — женщина);
* EDUCATION — образование;
* MARITAL_STATUS — семейное положение;
* CHILD_TOTAL — количество детей клиента;
* DEPENDANTS — количество иждивенцев клиента;
* SOCSTATUS_WORK_FL — социальный статус клиента относительно работы (1 — работает, 0 — не работает);
* SOCSTATUS_PENS_FL — социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер);
* REG_ADDRESS_PROVINCE — область регистрации клиента;
* FACT_ADDRESS_PROVINCE — область фактического пребывания клиента;
* POSTAL_ADDRESS_PROVINCE — почтовый адрес области;
* FL_PRESENCE_FL — наличие в собственности квартиры (1 — есть, 0 — нет);
* OWN_AUTO — количество автомобилей в собственности.""")

st.markdown('Размерность таблицы:')
st.write(clients.shape)
st.markdown('Семпл данных:')
st.write(clients.head())

cmd = "find_missing_values(clients, True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')
cmd = "find_duplicates(clients, ['id'], True)"
st.code(f"{cmd}\n>>> {eval(cmd)}", language='python')

st.markdown("""Первый вопрос - зачем нам `work` и `pens`, если у нас уже есть эти признаки в таблице? Видимо не нужны.""")

age_counts = clients['age'].value_counts()
plt.figure(figsize=(10, 6))
age_counts.plot(kind='bar')
plt.title('Распределение возраста клиентов')
plt.xlabel('Age')
plt.ylabel('Frequency')
st.pyplot()
st.markdown("""Наблюдается красивая убывающая линия по возрасту клиентов.

Можно так еще много строить графиков, но лучше уже все поджойнить и смотреть в совокупности.""")

##########################################################################################
st.markdown("""### Джойним все таблички""")

cmd = """df = (
    pd.merge(clients, target.drop('agreement_rk', axis=1), left_on='id', right_on='id_client', how='left')
    .drop('id', axis=1)
)
df = pd.merge(df, job, how='left')
df = pd.merge(df, salary, how='left')
df = pd.merge(df, last_credit, how='left')
df = pd.merge(df, loan, how='left')
df = pd.merge(df, close_loan, how='left')"""
st.code(f"{cmd}", language='python')

df = (
    pd.merge(clients, target.drop('agreement_rk', axis=1), left_on='id', right_on='id_client', how='left')
    .drop('id', axis=1)
)
df = pd.merge(df, job, how='left')
df = pd.merge(df, salary, how='left')
df = pd.merge(df, last_credit, how='left')
df = pd.merge(df, loan, how='left')
df = pd.merge(df, close_loan, how='left')

st.markdown('Размерность таблицы:')
st.write(df.shape)
st.markdown('Семпл данных:')
st.write(df.head())

st.markdown("Посмотрим, как обычно, сначала на распределения.")

plt.figure(figsize=(15, 12))
numerical_columns = ['age', 'child_total', 'dependants', 'work_time', 'personal_income', 'credit', 'term', 'fst_payment']
categorical_columns = ['gender', 'education', 'marital_status', 'socstatus_work_fl', 'own_auto']

plots = len(numerical_columns)
cols = 3
rows = plots // cols + (plots % cols > 0)

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'Распределение: {column}')
plt.tight_layout()
st.pyplot()

plots = len(categorical_columns)
cols = 2
rows = plots // cols + (plots % cols > 0)
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(rows, cols, i)
    sns.countplot(y=column, data=df)
    plt.title(f'Частотности: {column}')
plt.tight_layout()
st.pyplot()

st.markdown("""К сожалению, уже потратил много времени на предыдущие датафреймы и немного не попадаю в дедлайн 🫠

Поэтому тут ускорюсь, покажу некоторые Data Quality проверки на бизнес-логику и отправлю в AnyTask.""")

st.markdown("""Проверим, есть ли клиенты с количеством детей больше, чем иждивенцев.""")
st.code("inconsistency_mask = df['child_total'] > df['dependants']", language='python')
inconsistency_mask = df['child_total'] > df['dependants']
inconsistent_data = df[inconsistency_mask]
st.write(f"Найдено данных: {inconsistent_data.shape[0]}")
st.markdown("Семпл данных:")
st.write(inconsistent_data[['child_total', 'dependants']].head())
st.markdown("Но тут на самом деле неоднозначно: дети больше 18 лет уже не иждивенцы, хотя все еще дети. Поэтому этот момент опустим, но может стоит дальше разобрать аналитику.")

st.markdown("""Другая потенциальная несогласованность - возраст человека и его стаж. Вычтем из возраста 18, умножим на 12 и сравним со стажем.""")
st.code("max_work_time = (df['age'] - 18) * 12\nlogical_inconsistency_mask = df['work_time'] > max_work_time", language='python')
max_work_time = (df['age'] - 18) * 12
logical_inconsistency_mask = df['work_time'] > max_work_time
logical_inconsistencies = df[logical_inconsistency_mask]
st.write(f"Количество таких семплов: {logical_inconsistencies.shape[0]}")
st.markdown("Семпл данных:")
logical_inconsistencies['had_start_to_work_from (years)'] = (logical_inconsistencies['age'] - logical_inconsistencies['work_time']/12).round(1)
st.write(logical_inconsistencies[['age', 'work_time', 'had_start_to_work_from (years)']])
st.markdown("Видно, что есть совсем несостыковки по типу начала работы в 10 лет, 0 или -13.")

st.markdown("""Думаю, что дальше можно еще копать, но у нас все же прикладной питон, а не дата анализ, поэтому остановлюсь.

**Поэтому на этом все, спасибо за внимание!**""")

