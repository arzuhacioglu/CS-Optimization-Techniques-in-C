/*
 * Project: CS Optimization Techniques - MNIST Binary Classification
 *
 * Description:
 * This program implements a Single-Layer Neural Network from scratch in C to classify
 * handwritten digits (specifically '4' and '8') from the MNIST dataset.
 *
 * Key Objectives:
 * 1. Optimization Algorithms: Custom implementation of Gradient Descent (GD),
 * Stochastic Gradient Descent (SGD), and ADAM (Adaptive Moment Estimation).
 * 2. Mathematical Foundation: Computes gradients via Backpropagation using the Chain Rule
 * without relying on high-level ML libraries like TensorFlow or PyTorch.
 * 3. Efficiency: Manual memory management and matrix operations for performance analysis.
 *
 * Technical Details:
 * - Activation Function: tanh(x)
 * - Loss Function: Mean Squared Error (MSE)
 * - Data Parsing: Reads raw binary IDX format directly.
 *
 * Usage:
 * Compile: gcc mnist_optimizer.c -o optimizer -lm
 * Run: ./optimizer
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/stat.h>  
#include <direct.h>


#define IMAGE_SIZE 784   // Her resim 28x28 = 784 piksel
#define NUM_IMAGES 80    // Her rakam icin resim sayisi(4 ve 8)
#define SIZE (IMAGE_SIZE + 1) // 784 piksel + 1 bias
#define TRAIN_NUM 160 // Toplam egitim ornegi sayfasi
#define NUM_TESTIMAGES 20   
#define TOTAL_TESTIMAGES 40
#define MAX_ITER 100 

// Verileri tutacak yapi
typedef struct {
    double loss;
    double time_spent;
    double accuracy;
    double test_accur;    
	double *weights;
} IterationResult;

// Global sonuc dizisi
IterationResult gdegitimsonuc[MAX_ITER];
IterationResult sgdegitimsonuc[MAX_ITER];
IterationResult adamegitimsonuc[MAX_ITER];

// Fonksiyon Prototipleri
void create_directory_if_not_exists(const char *path);
void read_and_normalize_idx_file(const char*, const char*, double***);
void read_and_normalize_idx_testfile(const char*, const char*, double***);
void make_y(double*);
void make_ytest(double*);
double w_x(double*, double*);
double loss(double*, double**, double*, int);
void gradient_descent(double **, double *, double *, double**, double*);
void stochastic_gradient_descent(double **, double *, double *, double**, double*);
void adam(double **, double *, double *, double**, double*);
void save_results_to_csv(const char *, IterationResult *);
double test(double**, double*, double*);

int main() {
    int i, j;
    double eps = 0.01, accuracy;

    // Dosya yollari
    // Dosya yollar�
    const char *filename_4 = "train_4_images.idx";
    const char *filename_8 = "train_8_images.idx";
    const char *test_filename_4 = "test_4_images.idx";
    const char *test_filename_8 = "test_8_images.idx";

    // Goruntu ve test matrisini okuma ve normalizasyon
    double **images_matrix = NULL;
    read_and_normalize_idx_file(filename_4, filename_8, &images_matrix);
    double **test_matrix = NULL;
    read_and_normalize_idx_testfile(test_filename_4, test_filename_8, &test_matrix);
    
    // Y matrisini olustur
    double y[TRAIN_NUM];
    make_y(y);
    double y_testmatrix[TOTAL_TESTIMAGES];
    make_ytest(y_testmatrix); 
    

    //agirlik vektorunu baslat
    double *w = (double*)malloc(SIZE * sizeof(double));
    if (w == NULL) {
        printf("Bellek tahsisi hatasi");
        exit(1);
    }
    
    srand(time(NULL));

    //ilk w degerini belirleme
    w[0] = 0.1;

  /*  // Gradient Descent islemini baslat
    gradient_descent(images_matrix, y, w, test_matrix, y_testmatrix);

	const char *directory = "gdegitimsonuc.csv";
    
    save_results_to_csv(directory, gdegitimsonuc);

*/
    // SGD islemini baslat
    stochastic_gradient_descent(images_matrix, y, w, test_matrix, y_testmatrix);

	const char *directory2 = "sgdegitimsonuc.csv"; 
    
    save_results_to_csv(directory2, sgdegitimsonuc);

 /*   //adam gibi adami baslat
    adam(images_matrix, y, w, test_matrix, y_testmatrix);

    const char *directory3 = "adamegitimsonuc.csv"; 
    
    save_results_to_csv(directory3, adamegitimsonuc);
*/

    // Bellek temizleme
    for (i = 0; i < TRAIN_NUM; i++){
		free(images_matrix[i]);
    	}
	free(images_matrix);
	free(w);
	for(i = 0; i < SIZE; i++){
		free(gdegitimsonuc[i].weights);    
	}

    return 0;
}

// Dizin olusturma fonksiyonu
void create_directory_if_not_exists(const char *path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        if (mkdir(path) == -1) {
            perror("Dizin olusturulamadi");
            exit(EXIT_FAILURE);
        }
        printf("Dizin olusturuldu: %s\n", path);
    }
}

// Goruntuleri okuma ve normalizasyon
void read_and_normalize_idx_file(const char *filename_4, const char *filename_8, double ***images_matrix) {
    FILE *file_4 = fopen(filename_4, "rb");
    FILE *file_8 = fopen(filename_8, "rb");
    
    int i, j;

    if (file_4 == NULL || file_8 == NULL) {
        perror("Dosya acilamadi");
        exit(1);
    }

    uint8_t buffer[IMAGE_SIZE];
    *images_matrix = malloc(TRAIN_NUM * sizeof(double*));
    for (i = 0; i < TRAIN_NUM; i++) {
        (*images_matrix)[i] = malloc(SIZE * sizeof(double));
    }

    // 4'leri oku ve normalize et
    for (i = 0; i < NUM_IMAGES; i++) {
        fread(buffer, sizeof(uint8_t), IMAGE_SIZE, file_4);
        for (j = 0; j < IMAGE_SIZE; j++) {
            (*images_matrix)[i][j] = buffer[j] / 255.0; 
        }
        (*images_matrix)[i][IMAGE_SIZE] = 1.0;
    }

    // 8'leri oku ve normalize et
    for (i = NUM_IMAGES; i < TRAIN_NUM; i++) {
        fread(buffer, sizeof(uint8_t), IMAGE_SIZE, file_8);
        for (j = 0; j < IMAGE_SIZE; j++) {
            (*images_matrix)[i][j] = buffer[j] / 255.0; 
        }
        (*images_matrix)[i][IMAGE_SIZE] = 1.0;
    }

    fclose(file_4);
    fclose(file_8);
}

//test verilerini okuma ve normallestirme
void read_and_normalize_idx_testfile(const char *test_filename_4, const char *test_filename_8, double ***test_matrix) {
    FILE *file_4 = fopen(test_filename_4, "rb");
    FILE *file_8 = fopen(test_filename_8, "rb");
    
    int i, j;

    if (file_4 == NULL || file_8 == NULL) {
        perror("Dosya acilamadi");
        exit(1);
    }

    uint8_t buffer[IMAGE_SIZE];
    *test_matrix = malloc(TOTAL_TESTIMAGES * sizeof(double*));
    for (i = 0; i < TOTAL_TESTIMAGES; i++) {
        (*test_matrix)[i] = malloc(IMAGE_SIZE * sizeof(double));
    }
    
    // 4'leri oku ve normalize et
    for (i = 0; i < NUM_TESTIMAGES; i++) {
        fread(buffer, sizeof(uint8_t), IMAGE_SIZE, file_4);
        for (j = 0; j < IMAGE_SIZE; j++) {
            (*test_matrix)[i][j] = buffer[j] / 255.0; 
        }
        (*test_matrix)[i][IMAGE_SIZE] = 1.0;
    }

    // 8'leri oku ve normalize et
    for (i = NUM_TESTIMAGES; i < TOTAL_TESTIMAGES; i++) {
        fread(buffer, sizeof(uint8_t), IMAGE_SIZE, file_8);
        for (j = 0; j < IMAGE_SIZE; j++) {
            (*test_matrix)[i][j] = buffer[j] / 255.0; 
        }
        (*test_matrix)[i][IMAGE_SIZE] = 1.0;
    }

    fclose(file_4);
    fclose(file_8);
}

// y matrisini dolduran fonksiyon
void make_y(double *y) {
	int i;
    for (i = 0; i < NUM_IMAGES; i++) y[i] = 1.0;
    for (i = NUM_IMAGES; i < TRAIN_NUM; i++) y[i] = -1.0;
}

// test y matrisini dolduran fonksiyon
void make_ytest(double *y) {
	int i;
    for (i = 0; i < NUM_TESTIMAGES; i++) y[i] = 1.0;
    for (i = NUM_TESTIMAGES; i < TOTAL_TESTIMAGES; i++) y[i] = -1.0;
}


// tanh icin agirliklari toplam
double w_x(double *w, double *x) {
	int i;
    double sum = 0.0;
    for (i = 0; i < SIZE; i++) {
        sum += w[i] * x[i];
    }
    return sum;
}

// kayip fonksiyonu
double loss(double *w, double **x, double *y, int tane) {
	int i;
    double total_loss = 0.0;
    for (i = 0; i < tane; i++) {
        double y_tahmin = tanh(w_x(w, x[i]));
        total_loss += (y_tahmin - y[i]) * (y_tahmin - y[i]);
    }
    return total_loss / tane;
}

double calculate_accuracy(double *w, double **x, double *y, int tane) {
    int i;
	double correct = 0.0f;
    for (i = 0; i < tane; i++) {
        double y_tahmin = tanh(w_x(w, x[i]));
        if ((y_tahmin >= 0.5 && y[i] == 1) || (y_tahmin < -0.5 && y[i] == -1)) {
            correct++;
        }
    }
    return correct / tane;
}

void gradient_descent(double **images_matrix, double *y, double *w, double** test_matrix, double* y_testmatrix) {

	int iter, i, j;
	double eps = 0.01;
    double *toplam_turev = (double*)calloc(SIZE, sizeof(double));
    double *turev = (double*)calloc(SIZE, sizeof(double));
    
    if (toplam_turev == NULL || turev == NULL) {
        printf("Bellek tahsisi hatas�\n");
        exit(1);
    }
    
    clock_t start_time, end_time;
    double time_spent;
    
    
    for (iter = 0; iter < MAX_ITER; iter++) {
        start_time = clock();
        
        for (j = 0; j < SIZE; j++) toplam_turev[j] = 0;

        for (i = 0; i < TRAIN_NUM; i++) {
            double y_tahmin = tanh(w_x(w, images_matrix[i]));
            for (j = 0; j < SIZE; j++) {
                turev[j] = 2 * (y_tahmin - y[i]) * (1 - y_tahmin * y_tahmin) * images_matrix[i][j];
                toplam_turev[j] += turev[j];
            }
        }

        // agirliklari guncelle
        for (j = 0; j < SIZE; j++) {
            w[j] -= eps * toplam_turev[j] / TRAIN_NUM;
        }

        // Kayip hesapla ve sonucu kaydet
        double L = loss(w, images_matrix, y, TRAIN_NUM);
        double accuracy = calculate_accuracy(w, images_matrix, y, TRAIN_NUM);
    	
    	double test_accur = test(test_matrix, y_testmatrix, w);
    	
        end_time = clock();
        time_spent += (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;

        gdegitimsonuc[iter].loss = L;
        gdegitimsonuc[iter].time_spent = time_spent;
        gdegitimsonuc[iter].accuracy = accuracy;
        gdegitimsonuc[iter].test_accur = test_accur;
        gdegitimsonuc[iter].weights = (double *)malloc(SIZE * sizeof(double));
        for (j = 0; j < SIZE; j++) {
            gdegitimsonuc[iter].weights[j] = w[j];
        }

        printf("GD iterasyon %d - Kayip: %f - Zaman: %f - Dogruluk: %f\n - Test: %f\n", iter + 1, L, time_spent, accuracy, test_accur);
    }
	
    free(toplam_turev);
    free(turev);
}

//cocacolastic 
void stochastic_gradient_descent(double **images_matrix, double *y, double *w, double** test_matrix, double* y_testmatrix) {
    
	int iter,j,i;
    double eps = 0.01;
    double *turev = (double*)calloc(SIZE, sizeof(double));
    
    clock_t start_time, end_time;
    double time_spent;
    
    
    for (iter = 0; iter < MAX_ITER; iter++) {
    	
        start_time = clock();
        

        // Rastgele bir satir sec
        int randomRow = rand() % TRAIN_NUM;
        double y_tahmin = tanh(w_x(w, images_matrix[randomRow]));

        for (j = 0; j < SIZE; j++) {
            turev[j] = 2 * (y_tahmin - y[randomRow]) * (1 - y_tahmin * y_tahmin) * images_matrix[randomRow][j];
        }

        //agirliklari guncelle
        for (i = 0; i < SIZE; i++) {
            w[i] -= eps * turev[i];
        }

        // Kayip hesapla ve sonucu kaydet
        double L = loss(w, images_matrix, y, TRAIN_NUM);
        double accuracy = calculate_accuracy(w, images_matrix, y, TRAIN_NUM);
        
		double test_accur = test(test_matrix, y_testmatrix, w);
        
        end_time = clock();
        time_spent += (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;

        sgdegitimsonuc[iter].loss = L;
        sgdegitimsonuc[iter].time_spent = time_spent;
        sgdegitimsonuc[iter].accuracy = accuracy;
        sgdegitimsonuc[iter].test_accur = test_accur;
        sgdegitimsonuc[iter].weights = (double *)malloc(SIZE * sizeof(double));
        for (j = 0; j < SIZE; j++) {
        sgdegitimsonuc[iter].weights[j] = w[j];
        }

        printf("SGD iterasyon %d - Kayip: %f - Zaman: %f - Dogruluk: %f\n - Test: %f\n", iter + 1, L, time_spent, accuracy, test_accur);
    }

    free(turev);
}

//ADAM GİBİ ADAM KOCA ADAM
void adam(double **images_matrix, double *y, double *w, double** test_matrix, double* y_testmatrix) {
	
	int iter = 0, j;
    double B1 = 0.9, B2 = 0.999, E = pow(10, -8), eps = 0.01;
    double *turev = (double *)calloc(SIZE, sizeof(double));
    double *mt = (double *)calloc(SIZE, sizeof(double));
    double *vt = (double *)calloc(SIZE, sizeof(double));
    double *MT = (double *)calloc(SIZE, sizeof(double));
    double *VT = (double *)calloc(SIZE, sizeof(double));

    clock_t start_time, end_time;
    double time_spent;

	start_time = clock();
    
    
    while (iter < 100) {
    	

        iter++;

        int randomRow = rand() % TRAIN_NUM;
        double y_tahmin = tanh(w_x(w, images_matrix[randomRow]));

        for (j = 0; j < SIZE; j++) {
            turev[j] = 2 * (y_tahmin - y[randomRow]) * (1 - y_tahmin * y_tahmin) * images_matrix[randomRow][j];
            mt[j] = B1 * mt[j] + (1 - B1) * turev[j];
            vt[j] = B2 * vt[j] + (1 - B2) * pow(turev[j], 2);
            MT[j] = mt[j] / (1 - pow(B1, iter));
            VT[j] = vt[j] / (1 - pow(B2, iter));
            w[j] -= eps * (MT[j] / (sqrt(VT[j]) + E));
        }

        // Kayip hesapla ve sonucu kaydet
        double L = loss(w, images_matrix, y, TRAIN_NUM);
        double accuracy = calculate_accuracy(w, images_matrix, y, TRAIN_NUM);
        
        double test_accur = test(test_matrix, y_testmatrix, w);
        
        end_time = clock();
        time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;

        adamegitimsonuc[iter].loss = L;
        adamegitimsonuc[iter].time_spent = time_spent;
        adamegitimsonuc[iter].accuracy = accuracy;
        adamegitimsonuc[iter].test_accur = test_accur;
        adamegitimsonuc[iter].weights = (double *)malloc(SIZE * sizeof(double));
        for (j = 0; j < SIZE; j++) {
            adamegitimsonuc[iter].weights[j] = w[j];
        }

        printf("Adam iterasyon %d - Kayip: %f - Zaman: %f - Dogruluk: %f\n - Test: %f\n", iter, L, time_spent, accuracy, test_accur);
    }

    free(turev);
    free(mt);
    free(vt);
    free(MT);
    free(VT);
}

// Sonuclari belirtilen dosyaya kaydedecek fonksiyon
void save_results_to_csv(const char *directory, IterationResult *results) {
	int iter, i, j;
    // Dizin yoksa olustur
    create_directory_if_not_exists(directory);

    // Dosyayi ac
    FILE *file = fopen(directory, "w");
    if (file == NULL) {
        printf("Dosya acilamadi!\n");
        return;
    }

    // CSV basliklarini yaz
    fprintf(file, "Iterasyon,Kayip,Zaman,Dogruluk,Test Dogruluk");
    for (j = 0; j < SIZE; j++) {
        fprintf(file, ",Weight%d", j + 1); // Her ağırlık için sütun başlıkları
    }
    fprintf(file, "\n");

    // Sonuclari dosyaya yaz
    for (iter = 0; iter < MAX_ITER; iter++) {
        fprintf(file, "%d,%f,%f,%f,%f,", iter + 1, results[iter].loss, results[iter].time_spent, results[iter].accuracy, results[iter].test_accur);
        for (j = 0; j < SIZE; j++) {
            fprintf(file, "%f,", results[iter].weights[j]); // Ağırlıkları yaz
        }
        fprintf(file, "\n");
		fflush(file);    
	}
    
    // Dosyayi kapat
    fclose(file);
}

double test(double **test_matrix, double *y_testmatrix, double *w) {
    int correct = 0, i;
    double fark, y_tahmin;
    double accuracy;
	
    for(i=0 ; i < NUM_TESTIMAGES ; i++){
        y_tahmin = tanh(w_x(w, test_matrix[i]));
        fark = y_testmatrix[i] - y_tahmin;
        
        if(fark <= 0.5 && fark >= -0.5){
            correct++;
        }
	}
	
	accuracy = (double) correct/NUM_TESTIMAGES;
    
    return accuracy;
}


