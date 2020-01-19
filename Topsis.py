import numpy as np
import pandas as pd

def create_evaluation_matrix():
    df=pd.read_csv('data.csv')
    print(df.head())
    mat=df.values
    mat=mat[:,1:]# m rows n columns m alternatives and n criterias
    print(mat)
    create_normalised_matrix(mat)
def create_normalised_matrix(mat):
    print(mat.shape)
    column_squared_sum=np.zeros(mat.shape[1])# for each of the n features
    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            column_squared_sum[j]+=mat[i][j]*mat[i][j]
        column_squared_sum[j]=np.sqrt(column_squared_sum[j])
        mat[:,j:j+1]=mat[:,j:j+1]/column_squared_sum[j]
    print(column_squared_sum)
    print(mat)# this is the perfaprmance matrix
    weighted_normalised_matrix(mat,weight=np.asarray([1,2,2,1]))
def weighted_normalised_matrix( mat,weight):
    totalweight=np.sum(weight)
    weight=weight/totalweight
    weighted_normalised_mat=weight*mat
    calculate_ideal_best_and_ideal_worst(weighted_normalised_mat,is_max_the_most_desired=np.asarray([1,1,0,1]))
    # 1 mean max is ideal best and 0 is min value is the ideal best
def calculate_ideal_best_and_ideal_worst(weighted_normalised_mat,is_max_the_most_desired):
    print("******************************")
    print(weighted_normalised_mat)
    print("******************************")
    ideal_best=np.zeros(weighted_normalised_mat.shape[1])
    ideal_worst = np.zeros(weighted_normalised_mat.shape[1])
    for j in range(weighted_normalised_mat.shape[1]):
        if is_max_the_most_desired[j]==1:
            ideal_best[j]=np.max(weighted_normalised_mat[:,j])
            ideal_worst[j] = np.min(weighted_normalised_mat[:, j])
        else:
            ideal_worst[j] = np.max(weighted_normalised_mat[:, j])
            ideal_best[j] = np.min(weighted_normalised_mat[:, j])
    print(ideal_best)
    print(ideal_worst)
    euclidean_distance_from_ideal_best_and_ideal_worst_for_each_alternative(weighted_normalised_mat,ideal_best,ideal_worst)
def euclidean_distance_from_ideal_best_and_ideal_worst_for_each_alternative(mat, ideal_best,ideal_worst):
    euclidean_distance_from_ideal_best=np.zeros(mat.shape[0])
    euclidean_distance_from_ideal_worst=np.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        eachrowBest=0
        eachRowWorst=0
        for j in range(mat.shape[1]):
            eachrowBest+=(mat[i][j]-ideal_best[j])**2
            eachRowWorst+= (mat[i][j] - ideal_worst[j])**2
        euclidean_distance_from_ideal_best[i]=np.sqrt(eachrowBest)
        euclidean_distance_from_ideal_worst[i]=np.sqrt(eachRowWorst)
    print("###################")
    print(euclidean_distance_from_ideal_worst)
    print(euclidean_distance_from_ideal_best)
    print("""""""""""""""""""""'""""")
    performance_score(mat,euclidean_distance_from_ideal_best,euclidean_distance_from_ideal_worst)
def performance_score(mat,euclidean_best,euclidean_worst):
    performance=np.zeros(mat.shape[0])
    for i in range( mat.shape[0]):
        performance[i]=euclidean_worst[i]/(euclidean_best[i]+euclidean_worst[i])
    print("$$$$$$$$$$$$$$$$")
    print(performance)
    print("$$$$$$$$$$$$$$")
    l=list(performance)
    rank=[sorted(l,reverse=True).index(x) for x in l]
    print("perfrmance_score","rank",sep="       ")
    for i in range(mat.shape[0]):
        print(performance[i],rank[i]+1,sep="        ")




if __name__=='__main__':
    create_evaluation_matrix()
"""The TOPSIS process is carried out as follows:

Step 1
Create an evaluation matrix consisting of m alternatives and n criteria, with the intersection of each alternative and criteria given as {\displaystyle x_{ij}}x_{ij}, we therefore have a matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}}.
Step 2
The matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}} is then normalised to form the matrix
{\displaystyle R=(r_{ij})_{m\times n}}R=(r_{{ij}})_{{m\times n}}, using the normalisation method
{\displaystyle r_{ij}={\frac {x_{ij}}{\sqrt {\sum _{k=1}^{m}x_{kj}^{2}}}},\quad i=1,2,\ldots ,m,\quad j=1,2,\ldots ,n}{\displaystyle r_{ij}={\frac {x_{ij}}{\sqrt {\sum _{k=1}^{m}x_{kj}^{2}}}},\quad i=1,2,\ldots ,m,\quad j=1,2,\ldots ,n}
Step 3
Calculate the weighted normalised decision matrix
{\displaystyle t_{ij}=r_{ij}\cdot w_{j},\quad i=1,2,\ldots ,m,\quad j=1,2,\ldots ,n}{\displaystyle t_{ij}=r_{ij}\cdot w_{j},\quad i=1,2,\ldots ,m,\quad j=1,2,\ldots ,n}
where {\displaystyle w_{j}=W_{j}{\Big /}\sum _{k=1}^{n}W_{k},j=1,2,\ldots ,n}{\displaystyle w_{j}=W_{j}{\Big /}\sum _{k=1}^{n}W_{k},j=1,2,\ldots ,n} so that {\displaystyle \sum _{i=1}^{n}w_{i}=1}{\displaystyle \sum _{i=1}^{n}w_{i}=1}, and {\displaystyle W_{j}}W_{j} is the original weight given to the indicator {\displaystyle v_{j},\quad j=1,2,\ldots ,n.}{\displaystyle v_{j},\quad j=1,2,\ldots ,n.}
Step 4
Determine the worst alternative {\displaystyle (A_{w})}(A_{w}) and the best alternative {\displaystyle (A_{b})}(A_{b}):
{\displaystyle A_{w}=\{\langle \max(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{-}\rangle ,\langle \min(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{+}\rangle \rbrace \equiv \{t_{wj}\mid j=1,2,\ldots ,n\rbrace ,}{\displaystyle A_{w}=\{\langle \max(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{-}\rangle ,\langle \min(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{+}\rangle \rbrace \equiv \{t_{wj}\mid j=1,2,\ldots ,n\rbrace ,}
{\displaystyle A_{b}=\{\langle \min(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{-}\rangle ,\langle \max(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{+}\rangle \rbrace \equiv \{t_{bj}\mid j=1,2,\ldots ,n\rbrace ,}{\displaystyle A_{b}=\{\langle \min(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{-}\rangle ,\langle \max(t_{ij}\mid i=1,2,\ldots ,m)\mid j\in J_{+}\rangle \rbrace \equiv \{t_{bj}\mid j=1,2,\ldots ,n\rbrace ,}
where,
{\displaystyle J_{+}=\{j=1,2,\ldots ,n\mid j\}}{\displaystyle J_{+}=\{j=1,2,\ldots ,n\mid j\}} associated with the criteria having a positive impact, and
{\displaystyle J_{-}=\{j=1,2,\ldots ,n\mid j\}}{\displaystyle J_{-}=\{j=1,2,\ldots ,n\mid j\}} associated with the criteria having a negative impact.
Step 5
Calculate the L2-distance between the target alternative {\displaystyle i}i and the worst condition {\displaystyle A_{w}}A_{w}
{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{wj})^{2}}},\quad i=1,2,\ldots ,m,}{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{wj})^{2}}},\quad i=1,2,\ldots ,m,}
and the distance between the alternative {\displaystyle i}i and the best condition {\displaystyle A_{b}}A_b
{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{bj})^{2}}},\quad i=1,2,\ldots ,m}{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{bj})^{2}}},\quad i=1,2,\ldots ,m}
where {\displaystyle d_{iw}}d_{{iw}} and {\displaystyle d_{ib}}d_{{ib}} are L2-norm distances from the target alternative {\displaystyle i}i to the worst and best conditions, respectively.
Step 6
Calculate the similarity to the worst condition:
{\displaystyle s_{iw}=d_{iw}/(d_{iw}+d_{ib}),\quad 0\leq s_{iw}\leq 1,\quad i=1,2,\ldots ,m.}{\displaystyle s_{iw}=d_{iw}/(d_{iw}+d_{ib}),\quad 0\leq s_{iw}\leq 1,\quad i=1,2,\ldots ,m.}
{\displaystyle s_{iw}=1}s_{{iw}}=1 if and only if the alternative solution has the best condition; and
{\displaystyle s_{iw}=0}s_{{iw}}=0 if and only if the alternative solution has the worst condition.
Step 7
Rank the alternatives according to {\displaystyle s_{iw}\,\,(i=1,2,\ldots ,m).}{\displaystyle s_{iw}\,\,(i=1,2,\ldots ,m).}"""