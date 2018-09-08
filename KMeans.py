import numpy as np
import matplotlib.pyplot as plt
import random
         
class Cluster(object):
    """
        Cluster bevat vectoren en heeft een centroid
    """
    
    def __init__(self, id):
        self.id = id
        self.indexes = []
        self.centroid = None
                    
    def bereken_centroid(self, vector_lijst, dimensie):   
        """
            Bereken het gemiddelde van alle vectoren in het cluster
            
            @params vector_lijst        lijst met vectoren
            @param dimensie             dimensie
            
            @return centroid
        """ 
        vectoren     = vector_lijst[self.indexes][:,range(dimensie)]

        if len(vectoren) == 0:
            self.centroid = [0]*dimensie    # maak van centroid een 0 vector
        else: 
            self.centroid = vectoren.sum(axis=0) / len(vectoren)
        return self.centroid
    
    def get_kwadratische_fout(self, vector_lijst, dimensie):
        """
            Krijg de fout van het cluster
            
            @param vector_lijst             lijst met vectoren
            @param dimensie                 dimensie
            
            @return fout                    fout vector
        """

        fout = 0.0
        vectoren  = vector_lijst[self.indexes][:,range(dimensie)]
        
        for vector in vectoren:
            fout += self.get_afstand_tot_centroid(vector[0:dimensie]) ** 2
            
        return fout

    def get_afstand_tot_centroid(self, vector):
        """
            Bereken afstand van vector tot de centroid van dit cluster
            
            @param vector   vector
        """
        return np.linalg.norm(vector - self.centroid)
        
class KMeans(object):
    """
        Cluster algorithme KMeans. KMeans is een snel cluster algoritme maar heeft als vereiste het aantal clusters. 
        KMeans werkt door alle vectoren opnieuw in te delen (in de dichtstbijzijnde) totdat de fout (die elke stap afneemt) hetzelfde blijft. 
        De fout is de afstand van alle vectoren tot het centrum van het kluster waarin ze zijn ingedeeld
    """
    
    def __init__(self, vector_lijst, aantal_clusters = 9, dimensie = None):
        """
            @param aantal_clusters              hoeveelheid clusters
            @param dimensie                     dimensie
            @param vector_lijst                 lijst met vectoren
        """
        
        if dimensie is None:
            dimensie = len(vector_lijst[0])
            
        self.aantal_clusters = aantal_clusters
        self.vector_lijst = vector_lijst
        self.dimensie = dimensie

        self.clusters = [Cluster(id=i) for i in range(0, aantal_clusters)]
        self.labels   = []
        
        # lijst met fout vectoren per cluster
        self.fout = None
        self.vorige_fout = None
        
        self.begin_centroids = None
  
    def _cluster_vectoren_randomly(self):
        """
            Verdeel alle vectoren willekeurig over de clusters
        """
        for i in range(len(self.vector_lijst)):
            random_cluster_index = random.randint(0, self.aantal_clusters-1)
            self.clusters[random_cluster_index].indexes.append(i)
    
    def _bereken_centroids(self):
        """
            Bereken het middelpunt van elk cluster
        """
        for cluster in self.clusters:
            cluster.bereken_centroid(self.vector_lijst, self.dimensie)
    
    def _cluster_vectoren(self):
        """
            Deel alle vectoren opnieuw in, bij het cluster met het dichtstbijzijnde centrum
        """
        self._clear_clusters()
        self.labels = []
        
        for index, vector in enumerate(self.vector_lijst):   
            kleinste_afstand = None
            dichtstbijzijnde_clusters = []
            
            for cluster in self.clusters:
            
                afstand = cluster.get_afstand_tot_centroid(vector[0:self.dimensie])
                
                # kijk in welk cluster(s) de vector het best past
                if kleinste_afstand is None or afstand < kleinste_afstand:
                    kleinste_afstand = afstand
                    dichtstbijzijnde_clusters = [cluster]
                    
                elif afstand == kleinste_afstand:
                    dichtstbijzijnde_clusters.append(cluster)
            
            # kies randomly tussen clusters met een gelijke afstand
            aantal_matches = len(dichtstbijzijnde_clusters)
            dichtstbijzijnde_cluster = None

            if aantal_matches > 1:
                # hoe groter de data set hoe kleiner de kans dat er meerdere matches zijn
                dichtstbijzijnde_cluster = dichtstbijzijnde_clusters[random.randint(0, aantal_matches-1)]
            else:
                dichtstbijzijnde_cluster = dichtstbijzijnde_clusters[0]
            self.labels.append(dichtstbijzijnde_cluster.id)
            dichtstbijzijnde_cluster.indexes.append(index)
        
    def _is_error_kleiner_geworden(self):
        """
            Is de fout verder afgenomen of blijft hij hetzelfde?
            
            @return False wanneer de fout niet is afgenomen anders True
        """
        self.vorige_fout = self.fout  
        self.fout = []

        for cluster in self.clusters:
            self.fout.append(cluster.get_kwadratische_fout(self.vector_lijst, self.dimensie))    
        return not np.array_equal(self.vorige_fout, self.fout)
    
    def _clear_clusters(self):
        """
            Verwijder de vectoren uit alle clusters
        """
        for cluster in self.clusters:
            cluster.indexes = []

    def get_begin_centroids(self):
        """
            @return     begin centroids
        """
        return self.begin_centroids
        
    def get_centroids(self):
        """
            @return     begin centroids
        """
        centroids = []
        for cluster in self.clusters:
            centroids.append(cluster.centroid)

        return np.array(centroids)
        
    def _set_begin_centroids(self, centroids):
        """
            @param centroids    centroids
        """
        for i, cluster in enumerate(self.clusters):
                cluster.centroid = centroids[i]
        
        # deel de vectoren in zodat de fout van de clusters kan worden uitgereken
        self._cluster_vectoren()
      
    def cluster(self, centroids = None, stap_callback = None):
        """
            cluster vectoren
        
            @param centroids        begin centroids
            @param stap_callback    een functie die elke stap geroepen wordt  (bijvoorbeeld om het per stap the plotten)
                                    met params (self) 
            
            @return self
        """
        self._clear_clusters()
        
        if centroids is None:
            self._cluster_vectoren_randomly()           # Deel clusters random in wanneer de begin centroids niet gespecificeerd zijn
            self._bereken_centroids()
        else:
            for i in range(len(self.clusters)):
                self.clusters[i].centroid = centroids[i]
            self._cluster_vectoren()                    # deel de vectoren in zodat de fout van de clusters kan worden uitgerekend
        
        # Sla begin centroids op omdat begin centroids altijd leiden tot dezelfde clusters
        self.begin_centroids = np.zeros([self.aantal_clusters, self.dimensie])
        
        for i, cluster in enumerate(self.clusters):
            self.begin_centroids[i] = cluster.centroid
        
        iteration = 0
        
        while self._is_error_kleiner_geworden():        # blijf clusteren zolang de kwadratische fout kleiner blijft worden
                      
            self._bereken_centroids()                   # bereken middelpunten
            self._cluster_vectoren()                    # deel alle vectoren opnieuw in
 
            if stap_callback is not None:
                stap_callback(self, iteration)
                
            iteration += 1
        
        return self

import math

"""
demo's

"""
  
def demo_kmeans_plot():
    def plot_kmeans(kmeans, iteration):
        labels = kmeans.labels
        data   = kmeans.vector_lijst
        centroids = kmeans.get_centroids()
        
        plt.clf()
        
        plot_labels_2d(data, labels)            
        plt.plot(centroids[:,0], centroids[:,1], marker="+", c="k", linestyle='None', markersize=10.0)
        plt.pause(0.00001)
	
    data = np.random.rand(200, 3); #INSERT DATA HERE
    kmeans      = KMeans(aantal_clusters = 5, vector_lijst = data, dimensie = 2)
    kmeans.cluster(stap_callback=plot_kmeans)
    kmeans.get_centroids()
    
""""
Help functies

"""

def get_best_kmeans_clustering(kmeans, repetitions = 100, print_progress=False):
        """
            Zoek de centroids die de kleinste fout geven
        
            @param  repetitions     aantal herhalingen
            @param  print_progress  print vooruitgang
            
            @return centroids           die de kleinste fout geven
                    smallest_fout_sum   som van de fout
        """
        
        best_centroids = None
        smallest_fout_sum = None

        for i in range(repetitions):
            kmeans.cluster(centroids=None)
            fout_sum = sum(kmeans.fout)
            
            if print_progress:
                print(" {:<4s}      {:s}".format(str(i), str(fout_sum)))
            
            if smallest_fout_sum is None or fout_sum < smallest_fout_sum:
                smallest_fout_sum = fout_sum
                best_centroids = kmeans.get_begin_centroids()
            
        if print_progress:
            print("\nSmallest fout_sum : "+str(smallest_fout_sum))
            print(best_centroids)
        return best_centroids
     
def plot_labels_2d(data, labels):
    cluster_ids = np.unique(labels)
    for id in cluster_ids:
        vectoren = data[np.where(labels == id)] 
        plt.plot(vectoren[:,0],vectoren[:,1], "*", linestyle='None')     
    
def to_output_file(data, labels, seperator= "   ", filename = "output.txt"):
    """
        Schrijft cluster naar file
    """
    inf = open(filename, "w+")
    
    cluster_ids = np.unique(labels)
    for id in cluster_ids:
        cluster_data = data[np.where(labels == id)] 
        
        for item in cluster_data:
            inf.write(str(item) +seperator+ str(id)+"\n")
    inf.close()

if __name__ == "__main__":
	demo_kmeans_plot()
