import laspy as lp
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.impute import KNNImputer

def abc(filelist):
    for filename in filelist:
        # a) wizualizację oryginalnych chmur punktów
        las = lp.read(filename)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        p_cloud = o3d.geometry.PointCloud()
        p_cloud.points = o3d.utility.Vector3dVector(points)
        p_cloud.colors = o3d.utility.Vector3dVector(colors / (256 * 256))
        o3d.visualization.draw_geometries([p_cloud])

        # b) automatyczne usuwanie punktów odstających metodą knn
        f_pcd, ind = p_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=4.0)

        #outliers = p_cloud.select_by_index(ind, invert=True)
        #outliers.paint_uniform_color([1, 0, 0])
        #o3d.visualization.draw_geometries([f_pcd, outliers])

        # c) wizualizację chmur po realizacji p. b)
        o3d.visualization.draw_geometries([f_pcd])


def de(path,name,filelist, cells_w=5, cells_h=5, classify=False):
    # d) wizualizację i zapis do pliku NMPT o rozdzielczości ok. 5m
    # e) wizualizację i zapis do pliku NMT o rozdzielczości ok. 5m
    nmptlist = []
    i = 0
    for filename in filelist:
        i += 1
        las = lp.read(filename)
        las_x, las_y = np.asarray(las.x), np.asarray(las.y)
        z = np.asarray(las.z)
        if classify:
            las_x = las_x[las.classification == 2]
            las_y = las_y[las.classification == 2]
            z = z[las.classification == 2]
        ncells_X = int((las.header.maxs[0] - las.header.mins[0]) / cells_w)
        ncells_Y = int((las.header.maxs[1] - las.header.mins[1]) / cells_h)
        nmpt = np.zeros((ncells_Y, ncells_X))
        for x in range(ncells_X):
            start_x = las.header.mins[0] + x * cells_w
            end_x = las.header.mins[0] + (x + 1) * cells_w
            for y in range(ncells_Y):
                start_y = las.header.mins[1] + y * cells_h
                end_y = las.header.mins[1] + (y + 1) * cells_h
                selected_x = (las_x >= start_x) & (las_x <= end_x)
                selected_y = (las_y >= start_y) & (las_y <= end_y)
                selected_ind = np.where(selected_x & selected_y)
                cells_z = z[selected_ind]
                z_mean = np.mean(cells_z)
                nmpt[y, x] = z_mean
        nmpt = np.flipud(nmpt)
        imputer = KNNImputer(n_neighbors=9, weights='distance')
        nmpt = imputer.fit_transform(nmpt)
        nmptlist.append(nmpt)
    nmpt_combined= np.concatenate((nmptlist[0], nmptlist[1], nmptlist[2]), axis=1)

    plt.imshow(nmpt_combined, cmap='Greys')
    plt.colorbar()
    plt.savefig(os.path.join(path, name))
    plt.show()
    return nmpt_combined



def fghi(filelist):
    for filename in filelist:
        las = lp.read(filename)
        las.points = las.points[las.classification == 6]
        points = np.vstack((las.x, las.y, las.z)).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        f_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=4.0)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(f_pcd.cluster_dbscan(eps=4.5, min_points=500, print_progress=True))
            # g) zliczenie budynków
            max_labels = labels.max()
            print(f"Wykryto {max_labels + 1} budynków")
            # f) wizualizację eksponującą różnymi kolorami oddzielne budynki
            colors = plt.get_cmap("prism")(labels / max_labels)
            colors[labels < 0] = 0
            f_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([f_pcd])

        cubatures=[]
        heights=[]
        for building in range(max_labels):
            b_points = []
            for label in range(labels.size):
                if labels[label]==building:
                    b_points.append(list(points[label]))
            #print(building)
            b_points=np.asarray(b_points).transpose()
            min_x = min(b_points[0])
            max_x = max(b_points[0])
            min_y = min(b_points[1])
            max_y = max(b_points[1])
            min_z = min(b_points[2])
            max_z = max(b_points[2])
            dx = max_x - min_x
            dy = max_y - min_y
            dz = max_z - min_z
            cubatures.append(dx*dy*dz)
            heights.append(dz)
        # h) wizualizację histogramu wysokości budynków
        plt.hist(heights)
        plt.title("Histogram wysokości budynków ")
        plt.show()
        print(cubatures)


def j(path,filelist):
    nmpt=de(path,'NMPT50.png',filelist,cells_w=50,cells_h=50)
    nmt = de(path, 'NMT50.png', filelist, cells_w=50, cells_h=50,classify=True)
    # j) wizualizację i zapis rastra wysokości zabudowy o rozdzielczości ok. 50m
    builds=nmpt-nmt
    plt.imshow(builds, cmap='Greys')
    plt.colorbar()
    plt.savefig(os.path.join(path, 'BUILDS50.png'))
    plt.show()


def k(filelist):

    t=[]
    for filename in filelist:
        las = lp.read(filename)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        p_cloud = o3d.geometry.PointCloud()
        p_cloud.points = o3d.utility.Vector3dVector(points)
        p_cloud.colors = o3d.utility.Vector3dVector(colors / (256 * 256))
        #o3d.visualization.draw_geometries([p_cloud])

        f_pcd, ind = p_cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=4.0)
        # k) wizualizację modelu 3D.
        f_pcd.estimate_normals()
        nor = np.asarray(f_pcd.normals)

        #voxel_pcd = f_pcd.voxel_down_sample(voxel_size=0.5)
        #uni_pcd = f_pcd.uniform_down_sample(every_k_points=2)

        tin = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(f_pcd)

        t.append(tin)
    o3d.visualization.draw_geometries([t[0][0],t[1][0],t[2][0]], mesh_show_back_face=True)



if __name__ == '__main__':
    #ścieżki do danych
    filename1 = r'C:\Users\Oliwia\Desktop\pw\Foto\LAS1.las'
    filename2 = r'C:\Users\Oliwia\Desktop\pw\Foto\LAS2.las'
    filename3 = r'C:\Users\Oliwia\Desktop\pw\Foto\LAS3.las'

    filelist = [filename1,filename2,filename3]

    #ścieżka do folderu w którym zapisywane są wyniki
    path = 'C:\\Users\\Oliwia\\Desktop\\pw\\Foto\\p2wyniki'

    abc(filelist)

    de(path,'NMPT5.png',filelist)
    #de(path, 'NMT5.png', filelist,classify=True)

    fghi(filelist)

    j(path,filelist)

    k(filelist)








