#ifndef LAPLACIANZC_H
#define LAPLACIANZC_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class LaplacianZC {
private:
    cv::Mat laplace;
    int aperture; // Taille du noyau (ex: 3 ou 7)[cite: 31]

public:
    LaplacianZC() : aperture(3) {}

        void setAperture(int a) {
        aperture = a; 
    }

    int getAperture() const {
        return aperture; 
    }

    // Calcul du Laplacien sous forme de valeurs flottantes 32 bits (CV_32F)[cite: 31]
    cv::Mat computeLaplacian(const cv::Mat& image) {
        cv::Laplacian(image, laplace, CV_32F, aperture); 
            return laplace; 
    }

    // Normalisation pour affichage (ramène les valeurs signées vers [0, 255] avec neutre à 128)[cite: 32]
    cv::Mat getLaplacianImage(double scale = -1.0) {
       
        if (scale < 0) {
           
            double lapmin, lapmax;
            cv::minMaxLoc(laplace, &lapmin, &lapmax);
                scale = 127.0 / std::max(-lapmin, lapmax);
        }

        cv::Mat laplaceImage;
        laplace.convertTo(laplaceImage, CV_8U, scale, 128);
            return laplaceImage;
    }

    // Extraction des contours par passage par zéro (Zero-Crossings)[cite: 34]
    cv::Mat getZeroCrossings(cv::Mat laplaceInput) {
        
        cv::Mat signImage;
        // Binarisation selon le signe (positif vs négatif)[cite: 34]
        cv::threshold(laplaceInput, signImage, 0, 255, cv::THRESH_BINARY); 

            cv::Mat binary;
        signImage.convertTo(binary, CV_8U); 

            // Dilatation pour faire ressortir la frontière du signe[cite: 34]
            cv::Mat dilated;
        cv::dilate(binary, dilated, cv::Mat()); 

            // La différence extrait le passage par zéro exact[cite: 34]
            return dilated - binary; 
    }
};

#endif