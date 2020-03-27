import zipfile
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster
import os

with zipfile.ZipFile("imagenes.zip", 'r') as zip_ref:
    zip_ref.extractall()

lista_nombres = os.listdir("imagenes")

imagenes = np.zeros([100*100*3, len(lista_nombres)])
for i in range(len(lista_nombres)):
    imagenes[:,i] = plt.imread('imagenes/%s'%(lista_nombres[i])).flatten()

n_clusters = np.arange(1,21)
inertia = np.zeros(20)

for j in n_clusters:
    k_means = sklearn.cluster.KMeans(n_clusters=j).fit(imagenes.T)
    inertia[j-1] = k_means.inertia_
    
plt.figure()
plt.plot(n_clusters, inertia)
plt.xlabel('n clusters')
plt.ylabel('inertia')

arg_min_inertia = np.argmin(inertia[1:]-inertia[:-1]) + 1
mejor_cluster = n_clusters[arg_min_inertia]

plt.axvline(x = mejor_cluster, color="r", linestyle="--")
plt.savefig('inercia.png')


k_means = sklearn.cluster.KMeans(n_clusters=mejor_cluster).fit(imagenes.T)

centros = k_means.cluster_centers_

indices_imagenes_cercanas = np.zeros([5,mejor_cluster])

for k in range(mejor_cluster):
    n_arg_imagenes = np.arange(len(lista_nombres))[k_means.labels_ == k]
    imagenes_cluster = imagenes.T[n_arg_imagenes,:]
    distancia_euclidiana = np.sum((imagenes_cluster - centros[k])*(
            imagenes_cluster - centros[k]), axis=1)
    indices_imagenes_cercanas[:,k] =  n_arg_imagenes[np.argsort(
            distancia_euclidiana)][:5]
            
plt.figure(figsize =(5*6, 5*mejor_cluster))

for m in range(mejor_cluster):
    plt.subplot(mejor_cluster,6,m*6+1)
    plt.imshow(np.reshape(centros[m,:], (100, 100, 3)))
    plt.title('Centro del cluster %i'%m)
    
    for l in range(2,7):
        plt.subplot(mejor_cluster,6,(m)*6 +l)
        plt.imshow(np.reshape(imagenes[:,int(indices_imagenes_cercanas[l-2,m])],(100,100,3)))
        plt.title('Imagen %i, cluster %i'%(l-1,m))

plt.savefig('ejemplo_clases.png')