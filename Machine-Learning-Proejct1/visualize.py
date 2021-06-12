from IPython.display import display
from IPython.display import Math
from IPython.display import Latex

X= train[:5000].values # 从训练集把数据读进来

# PCA visualization
X_std = StandardScaler().fit_transform(X)
k = 5
pca = PCA(n_components=k) # k 为主成分个数，自定
pca.fit(X_std)
X_5d = pca.transform(X_std)
Target = target[:5000]
pca_trace = go.Scatter(
    x = X_5d[:,0],
    y = X_5d[:,1],
    mode = 'markers',
    text = Target,
    showlegend = False,
    marker = dict(
        size = 8,
        color = Target,
        colorscale ='Rainbow',
        showscale = False,
        line = dict(width = 2,color = 'rgb(255, 255, 255)'),
        opacity = 0.8
    )
)
data = [pca_trace]
layout = go.Layout(
    title= 'MNIST 主成分分析(Principal Component Analysis)',
    hovermode= 'closest',
    xaxis= dict(title= 'Principal Component 1',ticklen= 5,zeroline= False,gridwidth= 2,),
    yaxis=dict(title= 'Principal Component 2',ticklen= 5,gridwidth= 2,),
    showlegend= True
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='pca-vis')

################################
# LDA visualization
lda = LDA(n_components=k)
X_LDA_2D = lda.fit_transform(X_std, Target.values )
traceLDA = go.Scatter(
    x = X_LDA_2D[:,0],
    y = X_LDA_2D[:,1],
    mode = 'markers',
    text = Target,
    showlegend = True,
    marker = dict(size = 8, color = Target,colorscale ='YlOrRd',showscale = False,
        line = dict(width = 2,color = 'rgb(255, 255, 255)'),
        opacity = 0.8
    )
)
data = [traceLDA]
layout = go.Layout(
    title= 'MNIST Linear Discriminant Analysis (LDA)',
    hovermode= 'closest',
    xaxis= dict(title= 'Linear Discriminant 1',ticklen= 4,zeroline= False,gridwidth= 1,
    ),
    yaxis=dict(title= 'Linear Discriminant 2',ticklen= 4,gridwidth= 1,
    ),
    showlegend= False
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='lda-vis')


################################
# T-sne visualization
n = 2
tsne = TSNE(n_components=n)
tsne_results = tsne.fit_transform(X_std) 
traceTSNE = go.Scatter(
    x = tsne_results[:,0],
    y = tsne_results[:,1],
    mode = 'markers',
    text = Target,
    showlegend = True,
    marker = dict(size = 8,color = Target,colorscale ='Rainbow',showscale = False,
        line = dict(width = 2,color = 'rgb(255, 255, 255)'),
        opacity = 0.8
    )
)
data = [traceTSNE]

layout = dict(title = 'MNIST TSNE (T-Distributed Stochastic Neighbour Embedding)',hovermode= 'closest',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False),
              showlegend= False,)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='tsne-vis')