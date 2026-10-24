#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Classe du cours pour le Laplacien et les passages par zero
#include "laplacianZC.h"

namespace fs = std::filesystem;

// ============================================================================
// STRUCTURE POUR STOCKER LES MESURES DEMANDEES DANS LE TP
// ============================================================================
struct Mesures {
    int contours_detectes = 0;   // Nombre de pixels contours dans l'image calculee
    int contours_reference = 0;  // Nombre de pixels contours dans la reference
    int contours_corrects = 0;   // Correctement detectes (TP)
    int faux_positifs = 0;       // Detectes mais non contours dans la reference (FP)
    int faux_negatifs = 0;       // Non detectes mais contours dans la reference (FN)

    double P = 0.0;              // Performance = corrects / (corrects + FP + FN)
    double TFP = 0.0;            // Taux de faux positifs = FP / (corrects + FP + FN)
    double TFN = 0.0;            // Taux de faux negatifs = FN / (corrects + FP + FN)
};

// ============================================================================
// 1. DETECTEUR DE SOBEL (Base sur derivatives.cpp du cours)
// ============================================================================
cv::Mat detecterSobel(const cv::Mat& image, int seuil, int tailleNoyau = 3, int tailleFlou = 3) {
    cv::Mat imgFloue;
    if (tailleFlou > 1 && (tailleFlou % 2 == 1)) {
        cv::GaussianBlur(image, imgFloue, cv::Size(tailleFlou, tailleFlou), 0);
    } else {
        imgFloue = image.clone();
    }

    // Calcul du gradient horizontal et vertical en 16-bits signe (CV_16S)
    cv::Mat sobelX, sobelY;
    cv::Sobel(imgFloue, sobelX, CV_16S, 1, 0, tailleNoyau);
    cv::Sobel(imgFloue, sobelY, CV_16S, 0, 1, tailleNoyau);

    // Approximation de l'amplitude totale : |SobelX| + |SobelY| comme dans le cours
    cv::Mat sobelTotal = cv::abs(sobelX) + cv::abs(sobelY);

    // Conversion en 8-bits pour affichage et binarisation
    cv::Mat sobel8U;
    sobelTotal.convertTo(sobel8U, CV_8U);

    // Binarisation par seuillage
    cv::Mat contours;
    cv::threshold(sobel8U, contours, seuil, 255, cv::THRESH_BINARY);
    return contours;
}

// ============================================================================
// 2. DETECTEUR LAPLACIEN (Base sur laplacianZC.h et derivatives1.cpp du cours)
// ============================================================================
cv::Mat detecterLaplace(const cv::Mat& image, int seuil, int tailleNoyau = 3, int tailleFlou = 3, bool modeZeroCrossing = true) {
    cv::Mat imgFloue;
    if (tailleFlou > 1 && (tailleFlou % 2 == 1)) {
        cv::GaussianBlur(image, imgFloue, cv::Size(tailleFlou, tailleFlou), 0);
    } else {
        imgFloue = image.clone();
    }

    // Utilisation de la classe LaplacianZC vue en cours
    LaplacianZC laplacien;
    laplacien.setAperture(tailleNoyau);

    cv::Mat laplaceF = laplacien.computeLaplacian(imgFloue);

    if (modeZeroCrossing) {
        // Extraction des contours par passage par zero (Zero-Crossing)
        cv::Mat zeroCrossings = laplacien.getZeroCrossings(laplaceF);

        // On filtre avec le seuil pour eviter de garder les faux passages par zero dans le bruit
        if (seuil > 0) {
            cv::Mat absLap;
            cv::convertScaleAbs(laplaceF, absLap);
            cv::Mat masque;
            cv::threshold(absLap, masque, seuil, 255, cv::THRESH_BINARY);
            cv::bitwise_and(zeroCrossings, masque, zeroCrossings);
        }
        return zeroCrossings;
    } else {
        // Seuil direct sur l'amplitude absolue du Laplacien
        cv::Mat absLap;
        cv::convertScaleAbs(laplaceF, absLap);
        cv::Mat contours;
        cv::threshold(absLap, contours, seuil, 255, cv::THRESH_BINARY);
        return contours;
    }
}

// ============================================================================
// 3. DETECTEUR DE CANNY (Base sur contours1.cpp du cours)
// ============================================================================
cv::Mat detecterCanny(const cv::Mat& image, int seuilBas, int seuilHaut, int aperture = 3, int tailleFlou = 1) {
    cv::Mat imgFloue;
    if (tailleFlou > 1 && (tailleFlou % 2 == 1)) {
        cv::GaussianBlur(image, imgFloue, cv::Size(tailleFlou, tailleFlou), 0);
    } else {
        imgFloue = image.clone();
    }

    cv::Mat contours;
    cv::Canny(imgFloue, contours, seuilBas, seuilHaut, aperture, true);
    return contours;
}

// ============================================================================
// 4. EVALUATION QUANTITATIVE (Formules du sujet + Option Voisinage 3x3)
// ============================================================================
Mesures evaluerContours(const cv::Mat& contoursDetectes, const cv::Mat& contoursRef, bool avecVoisinage3x3 = true) {
    Mesures m;

    m.contours_detectes = cv::countNonZero(contoursDetectes);
    m.contours_reference = cv::countNonZero(contoursRef);

    if (m.contours_detectes == 0 && m.contours_reference == 0) {
        m.P = 1.0;
        return m;
    }

    if (!avecVoisinage3x3) {
        // 1. Mode Exact : intersection stricte pixel a pixel
        cv::Mat intersectionMat;
        cv::bitwise_and(contoursDetectes, contoursRef, intersectionMat);
        m.contours_corrects = cv::countNonZero(intersectionMat);

        m.faux_positifs = m.contours_detectes - m.contours_corrects;
        m.faux_negatifs = m.contours_reference - m.contours_corrects;
    } else {
        // 2. Mode Voisinage 3x3 (recommande dans le sujet pour compenser le trace humain)
        // Dilatation 3x3 pour tolerer un ecart de 1 pixel
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat refDilatee, detDilatee;
        cv::dilate(contoursRef, refDilatee, element);
        cv::dilate(contoursDetectes, detDilatee, element);

        // Contours detectes qui tombent dans le voisinage 3x3 de la reference
        cv::Mat correctMat;
        cv::bitwise_and(contoursDetectes, refDilatee, correctMat);
        m.contours_corrects = cv::countNonZero(correctMat);

        // Faux positifs = detectes hors du voisinage de la reference
        m.faux_positifs = m.contours_detectes - m.contours_corrects;

        // Faux negatifs = pixels de reference n'ayant aucun contour detecte a proximite
        cv::Mat refTrouve;
        cv::bitwise_and(contoursRef, detDilatee, refTrouve);
        m.faux_negatifs = m.contours_reference - cv::countNonZero(refTrouve);
    }

    // Calcul des 3 formules demandees
    double total = static_cast<double>(m.contours_corrects + m.faux_positifs + m.faux_negatifs);
    if (total > 0.0) {
        m.P = static_cast<double>(m.contours_corrects) / total;
        m.TFP = static_cast<double>(m.faux_positifs) / total;
        m.TFN = static_cast<double>(m.faux_negatifs) / total;
    }

    return m;
}

// ============================================================================
// VISUALISATION COLOREE (Vert = TP, Rouge = FP, Bleu = FN)
// ============================================================================
cv::Mat creerImageEvaluationCouleur(const cv::Mat& imageGris, const cv::Mat& det, const cv::Mat& ref, bool avecVoisinage3x3 = true) {
    cv::Mat affichage;
    cv::cvtColor(imageGris, affichage, cv::COLOR_GRAY2BGR);
    affichage = affichage * 0.45; // Assombrir un peu le fond pour bien voir les contours

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat refDilatee, detDilatee;
    if (avecVoisinage3x3) {
        cv::dilate(ref, refDilatee, element);
        cv::dilate(det, detDilatee, element);
    } else {
        refDilatee = ref;
        detDilatee = det;
    }

    // Masques TP, FP, FN
    cv::Mat masqueTP, masqueFP, masqueFN;
    cv::bitwise_and(det, refDilatee, masqueTP);   // Vert (Vrais positifs)
    cv::bitwise_and(det, ~refDilatee, masqueFP);  // Rouge (Faux positifs)
    cv::bitwise_and(ref, ~detDilatee, masqueFN);  // Bleu (Faux negatifs)

    affichage.setTo(cv::Scalar(0, 0, 255), masqueFP);    // Rouge
    affichage.setTo(cv::Scalar(255, 150, 0), masqueFN);  // Bleu clair
    affichage.setTo(cv::Scalar(0, 255, 0), masqueTP);    // Vert

    return affichage;
}

// ============================================================================
// VARIABLES GLOBALES POUR LES TRACKBARS ET L'INTERFACE
// ============================================================================
std::vector<std::string> gListeImages;
int gIndexImage = 0;
int gChoixDetecteur = 2;       // 0: Sobel, 1: Laplace, 2: Canny
int gModeVoisinage = 1;        // 0: Exact 1x1, 1: Voisinage 3x3

// Parametres communs et intuitifs
int gSeuilPrincipal = 100;     // Seuil Sobel / Seuil Laplace / Seuil Bas Canny
int gCannySeuilHaut = 220;     // Seuil Haut pour Canny
int gIndexNoyau = 1;           // 0:k1, 1:k3, 2:k5, 3:k7
int gTailleFlou = 3;           // Flou gaussien (1, 3, 5...)

const std::string FENETRE_CONTROLES = "Reglages (Trackbars)";
const std::string FENETRE_RESULTAT = "TP3 - Detection de contours & Evaluation";

void actualiserAffichage() {
    if (gListeImages.empty()) return;

    if (gIndexImage < 0) gIndexImage = 0;
    if (gIndexImage >= static_cast<int>(gListeImages.size())) {
        gIndexImage = static_cast<int>(gListeImages.size()) - 1;
    }

    std::string nomBase = gListeImages[gIndexImage];
    std::string cheminImage = "dataset/images/" + nomBase + ".pgm";
    std::string cheminGT = "dataset/images/gt/" + nomBase + "_gt_binary.pgm";

    cv::Mat image = cv::imread(cheminImage, cv::IMREAD_GRAYSCALE);
    cv::Mat gt = cv::imread(cheminGT, cv::IMREAD_GRAYSCALE);

    if (image.empty() || gt.empty()) {
        std::cerr << "Erreur : impossible de lire " << nomBase << std::endl;
        return;
    }

    // Dans la base de reference, les contours sont noirs (0) sur fond blanc (255)
    cv::Mat refContours = (gt == 0);

    // Calcul des contours selon le detecteur selectionne
    cv::Mat contoursCalcules;
    std::string nomDetecteur = "";

    const int ksizes[] = { 1, 3, 5, 7 };
    const int cannyApertures[] = { 3, 5, 7 };

    int ksize = ksizes[gIndexNoyau];
    int cannyAp = (gIndexNoyau < 3) ? cannyApertures[gIndexNoyau] : 3;

    if (gChoixDetecteur == 0) {
        nomDetecteur = "Sobel";
        contoursCalcules = detecterSobel(image, gSeuilPrincipal, ksize, gTailleFlou);
    } else if (gChoixDetecteur == 1) {
        nomDetecteur = "Laplace (Zero-Crossing)";
        contoursCalcules = detecterLaplace(image, gSeuilPrincipal, ksize, gTailleFlou, true);
    } else {
        nomDetecteur = "Canny";
        contoursCalcules = detecterCanny(image, gSeuilPrincipal, gCannySeuilHaut, cannyAp, gTailleFlou);
    }

    // Evaluation quantitative
    bool avec3x3 = (gModeVoisinage == 1);
    Mesures m = evaluerContours(contoursCalcules, refContours, avec3x3);

    // Creation de la vue composite 4 quadrants
    int h = image.rows;
    int w = image.cols;
    cv::Mat composite(h * 2, w * 2, CV_8UC3, cv::Scalar(40, 40, 40));

    cv::Mat bgrOriginal, bgrRef, bgrDetectes;
    cv::cvtColor(image, bgrOriginal, cv::COLOR_GRAY2BGR);
    cv::cvtColor(255 - refContours, bgrRef, cv::COLOR_GRAY2BGR);
    cv::cvtColor(contoursCalcules, bgrDetectes, cv::COLOR_GRAY2BGR);
    cv::Mat vueCouleur = creerImageEvaluationCouleur(image, contoursCalcules, refContours, avec3x3);

    bgrOriginal.copyTo(composite(cv::Rect(0, 0, w, h)));
    bgrRef.copyTo(composite(cv::Rect(w, 0, w, h)));
    bgrDetectes.copyTo(composite(cv::Rect(0, h, w, h)));
    vueCouleur.copyTo(composite(cv::Rect(w, h, w, h)));

    // Ajout des titres
    cv::putText(composite, "1. Image originale: " + nomBase, cv::Point(15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    cv::putText(composite, "2. Reference GT (" + std::to_string(m.contours_reference) + " px)", cv::Point(w + 15, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    cv::putText(composite, "3. Contours detectes (" + std::to_string(m.contours_detectes) + " px)", cv::Point(15, h + 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    cv::putText(composite, "4. Evaluation coloree [Vert:TP | Rouge:FP | Bleu:FN]", cv::Point(w + 15, h + 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);

    // Bandeau d'informations en bas
    int hauteurBandeau = 80;
    cv::Mat affichageFinal(composite.rows + hauteurBandeau, composite.cols, CV_8UC3, cv::Scalar(25, 25, 25));
    composite.copyTo(affichageFinal(cv::Rect(0, 0, composite.cols, composite.rows)));

    std::ostringstream ligne1, ligne2;
    ligne1 << "[" << (gIndexImage + 1) << "/" << gListeImages.size() << "] "
           << "Detecteur: " << nomDetecteur << " | Mode: " << (avec3x3 ? "Voisinage 3x3" : "Exact 1x1")
           << " | Detectes=" << m.contours_detectes << " | Reference=" << m.contours_reference;

    ligne2 << ">> MESURES: Performance (P) = " << std::fixed << std::setprecision(2) << (m.P * 100.0) << "%"
           << " | Taux FP = " << (m.TFP * 100.0) << "%"
           << " | Taux FN = " << (m.TFN * 100.0) << "%"
           << " (TP=" << m.contours_corrects << ", FP=" << m.faux_positifs << ", FN=" << m.faux_negatifs << ")";

    cv::putText(affichageFinal, ligne1.str(), cv::Point(20, composite.rows + 30), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 220, 255), 1);
    cv::putText(affichageFinal, ligne2.str(), cv::Point(20, composite.rows + 60), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1);

    cv::imshow(FENETRE_RESULTAT, affichageFinal);
}

// Fonction de rappel pour les trackbars
void onTrackbar(int, void*) {
    actualiserAffichage();
}

// ============================================================================
// MODE INTERACTIF AVEC TRACKBARS
// ============================================================================
void lancerModeInteractif() {
    // Fenetre de controle elargie pour que tout soit parfaitement lisible
    cv::namedWindow(FENETRE_CONTROLES, cv::WINDOW_NORMAL);
    cv::resizeWindow(FENETRE_CONTROLES, 500, 360);
    cv::moveWindow(FENETRE_CONTROLES, 30, 40);

    // Fenetre de resultat redimensionnee et placee a droite pour ne pas se chevaucher
    cv::namedWindow(FENETRE_RESULTAT, cv::WINDOW_NORMAL);
    cv::resizeWindow(FENETRE_RESULTAT, 800, 850);
    cv::moveWindow(FENETRE_RESULTAT, 550, 40);

    int maxImages = static_cast<int>(gListeImages.size()) - 1;

    // Trackbars claires, concises et jamais tronquees
    cv::createTrackbar("Image (0-39)", FENETRE_CONTROLES, &gIndexImage, maxImages, onTrackbar);
    cv::createTrackbar("Detecteur (0:Sob, 1:Lap, 2:Can)", FENETRE_CONTROLES, &gChoixDetecteur, 2, onTrackbar);
    cv::createTrackbar("Voisinage (0:Exact, 1:3x3)", FENETRE_CONTROLES, &gModeVoisinage, 1, onTrackbar);
    cv::createTrackbar("Seuil Principal (0-255)", FENETRE_CONTROLES, &gSeuilPrincipal, 255, onTrackbar);
    cv::createTrackbar("Canny Seuil Haut (0-500)", FENETRE_CONTROLES, &gCannySeuilHaut, 500, onTrackbar);
    cv::createTrackbar("Taille Noyau (0:1, 1:3, 2:5, 3:7)", FENETRE_CONTROLES, &gIndexNoyau, 3, onTrackbar);
    cv::createTrackbar("Flou Gaussien (1, 3, 5..)", FENETRE_CONTROLES, &gTailleFlou, 15, onTrackbar);

    actualiserAffichage();

    std::cout << "\n-------------------------------------------------------" << std::endl;
    std::cout << " MODE INTERACTIF ACTIF" << std::endl;
    std::cout << " Raccourcis :" << std::endl;
    std::cout << "   [n] / [p] : Image Suivante / Precedente" << std::endl;
    std::cout << "   [d]       : Changer de detecteur" << std::endl;
    std::cout << "   [v]       : Alterner Voisinage 3x3 / Exact 1x1" << std::endl;
    std::cout << "   [b]       : Lancer le traitement par lot (Batch 40 images)" << std::endl;
    std::cout << "   [q] / ESC : Quitter" << std::endl;
    std::cout << "-------------------------------------------------------\n" << std::endl;

    while (true) {
        int touche = cv::waitKey(30);
        if (touche == 27 || touche == 'q' || touche == 'Q') {
            break;
        } else if (touche == 'n' || touche == 'N') {
            gIndexImage = (gIndexImage + 1) % gListeImages.size();
            cv::setTrackbarPos("Image (0-39)", FENETRE_CONTROLES, gIndexImage);
            actualiserAffichage();
        } else if (touche == 'p' || touche == 'P') {
            gIndexImage = static_cast<int>((gIndexImage - 1 + gListeImages.size()) % gListeImages.size());
            cv::setTrackbarPos("Image (0-39)", FENETRE_CONTROLES, gIndexImage);
            actualiserAffichage();
        } else if (touche == 'd' || touche == 'D') {
            gChoixDetecteur = (gChoixDetecteur + 1) % 3;
            cv::setTrackbarPos("Detecteur (0:Sob, 1:Lap, 2:Can)", FENETRE_CONTROLES, gChoixDetecteur);
            actualiserAffichage();
        } else if (touche == 'v' || touche == 'V') {
            gModeVoisinage = 1 - gModeVoisinage;
            cv::setTrackbarPos("Voisinage (0:Exact, 1:3x3)", FENETRE_CONTROLES, gModeVoisinage);
            actualiserAffichage();
        } else if (touche == 'b' || touche == 'B') {
            extern void lancerModeBatch();
            lancerModeBatch();
        }
    }

    cv::destroyAllWindows();
}

// ============================================================================
// MODE LOT D'IMAGES (BATCH SUR LES 40 IMAGES DU TP)
// ============================================================================
void lancerModeBatch() {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << " TRAITEMENT PAR LOT SUR LES " << gListeImages.size() << " IMAGES" << std::endl;
    std::cout << "=======================================================" << std::endl;

    std::ofstream fichierCSV("resultats_batch.csv");
    fichierCSV << "Image,Detecteur,Parametres,Contours_Detectes,Contours_Ref,"
               << "TP_Exact,FP_Exact,FN_Exact,P_Exact,TFP_Exact,TFN_Exact,"
               << "TP_3x3,FP_3x3,FN_3x3,P_3x3,TFP_3x3,TFN_3x3\n";

    struct ConfigTest {
        std::string nom;
        std::string params;
        std::vector<double> listeP_exact, listeP_3x3, listeTFP_3x3, listeTFN_3x3;
    };

    std::vector<ConfigTest> configs = {
        { "Sobel", "k=3, Seuil=80, Flou=3", {}, {}, {}, {} },
        { "Sobel (Haut Seuil)", "k=3, Seuil=130, Flou=3", {}, {}, {}, {} },
        { "Laplace (Zero-Crossing)", "k=3, Seuil=30, ZC=Oui", {}, {}, {}, {} },
        { "Laplace (Magnitude)", "k=3, Seuil=50, ZC=Non", {}, {}, {}, {} },
        { "Canny (Equilibre)", "T1=100, T2=220, Ap=3", {}, {}, {}, {} },
        { "Canny (Selectif)", "T1=150, T2=300, Ap=3", {}, {}, {}, {} }
    };

    for (size_t i = 0; i < gListeImages.size(); ++i) {
        std::string nomBase = gListeImages[i];
        std::string cheminImage = "dataset/images/" + nomBase + ".pgm";
        std::string cheminGT = "dataset/images/gt/" + nomBase + "_gt_binary.pgm";

        cv::Mat image = cv::imread(cheminImage, cv::IMREAD_GRAYSCALE);
        cv::Mat gt = cv::imread(cheminGT, cv::IMREAD_GRAYSCALE);
        if (image.empty() || gt.empty()) continue;

        cv::Mat refContours = (gt == 0);

        for (auto& cfg : configs) {
            cv::Mat det;
            if (cfg.nom == "Sobel") {
                det = detecterSobel(image, 80, 3, 3);
            } else if (cfg.nom == "Sobel (Haut Seuil)") {
                det = detecterSobel(image, 130, 3, 3);
            } else if (cfg.nom == "Laplace (Zero-Crossing)") {
                det = detecterLaplace(image, 30, 3, 3, true);
            } else if (cfg.nom == "Laplace (Magnitude)") {
                det = detecterLaplace(image, 50, 3, 3, false);
            } else if (cfg.nom == "Canny (Equilibre)") {
                det = detecterCanny(image, 100, 220, 3, 1);
            } else if (cfg.nom == "Canny (Selectif)") {
                det = detecterCanny(image, 150, 300, 3, 1);
            }

            Mesures mExact = evaluerContours(det, refContours, false);
            Mesures m3x3 = evaluerContours(det, refContours, true);

            cfg.listeP_exact.push_back(mExact.P);
            cfg.listeP_3x3.push_back(m3x3.P);
            cfg.listeTFP_3x3.push_back(m3x3.TFP);
            cfg.listeTFN_3x3.push_back(m3x3.TFN);

            fichierCSV << nomBase << "," << cfg.nom << ",\"" << cfg.params << "\","
                       << mExact.contours_detectes << "," << mExact.contours_reference << ","
                       << mExact.contours_corrects << "," << mExact.faux_positifs << "," << mExact.faux_negatifs << ","
                       << std::fixed << std::setprecision(4) << mExact.P << "," << mExact.TFP << "," << mExact.TFN << ","
                       << m3x3.contours_corrects << "," << m3x3.faux_positifs << "," << m3x3.faux_negatifs << ","
                       << m3x3.P << "," << m3x3.TFP << "," << m3x3.TFN << "\n";
        }
        std::cout << "\rTraitement image " << (i + 1) << " / " << gListeImages.size() << "..." << std::flush;
    }
    std::cout << " Termine !" << std::endl;
    fichierCSV.close();

    // Affichage des moyennes statistiques
    auto moyenne = [](const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        double s = 0.0;
        for (double x : v) s += x;
        return s / v.size();
    };

    auto ecartType = [](const std::vector<double>& v, double m) {
        if (v.empty()) return 0.0;
        double s = 0.0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / v.size());
    };

    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << " TABLEAU RECAPITULATIF DES MESURES (VOISINAGE 3x3 SUR LES 40 IMAGES)" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << std::left << std::setw(26) << "Detecteur"
              << std::setw(25) << "Parametres"
              << std::setw(18) << "Performance P"
              << std::setw(16) << "Taux FP"
              << std::setw(16) << "Taux FN" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& cfg : configs) {
        double mP = moyenne(cfg.listeP_3x3);
        double stdP = ecartType(cfg.listeP_3x3, mP);
        double mTFP = moyenne(cfg.listeTFP_3x3);
        double mTFN = moyenne(cfg.listeTFN_3x3);

        std::ostringstream pStr, tfpStr, tfnStr;
        pStr << std::fixed << std::setprecision(2) << (mP * 100.0) << "% +/- " << (stdP * 100.0) << "%";
        tfpStr << std::fixed << std::setprecision(2) << (mTFP * 100.0) << "%";
        tfnStr << std::fixed << std::setprecision(2) << (mTFN * 100.0) << "%";

        std::cout << std::left << std::setw(26) << cfg.nom
                  << std::setw(25) << cfg.params
                  << std::setw(18) << pStr.str()
                  << std::setw(16) << tfpStr.str()
                  << std::setw(16) << tfnStr.str() << std::endl;
    }

    std::cout << "\n==========================================================================================" << std::endl;
    std::cout << " COMPARAISON : CORRESPONDANCE EXACTE 1x1  VS  VOISINAGE 3x3" << std::endl;
    std::cout << "==========================================================================================" << std::endl;
    std::cout << std::left << std::setw(26) << "Detecteur"
              << std::setw(20) << "P (Exact 1x1)"
              << std::setw(20) << "P (Voisinage 3x3)"
              << std::setw(20) << "Gain Voisinage" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------" << std::endl;

    for (const auto& cfg : configs) {
        double mExact = moyenne(cfg.listeP_exact);
        double m3x3 = moyenne(cfg.listeP_3x3);
        double gain = (mExact > 0.0) ? (m3x3 / mExact) : 0.0;

        std::ostringstream exStr, v3Str, gainStr;
        exStr << std::fixed << std::setprecision(2) << (mExact * 100.0) << " %";
        v3Str << std::fixed << std::setprecision(2) << (m3x3 * 100.0) << " %";
        gainStr << "x " << std::fixed << std::setprecision(2) << gain;

        std::cout << std::left << std::setw(26) << cfg.nom
                  << std::setw(20) << exStr.str()
                  << std::setw(20) << v3Str.str()
                  << std::setw(20) << gainStr.str() << std::endl;
    }

    std::cout << "\n[Fichier exporte] -> resultats_batch.csv enregistre avec succes !" << std::endl;
    std::cout << "==========================================================================================\n" << std::endl;
}

// ============================================================================
// CHARGEMENT DE LA LISTE DES IMAGES DU DATASET
// ============================================================================
void chargerListeImages() {
    gListeImages.clear();
    std::string dossier = "dataset/images";
    if (!fs::exists(dossier)) {
        dossier = "../dataset/images";
    }
    if (!fs::exists(dossier)) {
        dossier = "TP3-Contours/dataset/images";
    }

    if (fs::exists(dossier)) {
        for (const auto& entry : fs::directory_iterator(dossier)) {
            if (entry.is_regular_file() && entry.path().extension() == ".pgm") {
                std::string nom = entry.path().stem().string();
                std::string gt = dossier + "/gt/" + nom + "_gt_binary.pgm";
                if (fs::exists(gt)) {
                    gListeImages.push_back(nom);
                }
            }
        }
        std::sort(gListeImages.begin(), gListeImages.end());
    }
}

// ============================================================================
// FONCTION PRINCIPALE
// ============================================================================
int main() {
    std::cout << "=======================================================" << std::endl;
    std::cout << " Polytech Marseille - Analyse d'Images (S7)" << std::endl;
    std::cout << " TP3 : Detection et Evaluation Quantitative de Contours" << std::endl;
    std::cout << "=======================================================" << std::endl;

    chargerListeImages();

    if (gListeImages.empty()) {
        std::cerr << "Erreur : dossier dataset/images introuvable !" << std::endl;
        std::cout << "Appuyez sur Entree pour quitter..." << std::endl;
        std::cin.get();
        return -1;
    }

    std::cout << "Nombre d'images chargees : " << gListeImages.size() << std::endl;
    std::cout << "\nMenu :" << std::endl;
    std::cout << "  1. Mode Interactif (Trackbars pour tester les parametres)" << std::endl;
    std::cout << "  2. Mode Lot d'images (Calcul automatique sur les 40 images)" << std::endl;
    std::cout << "  3. Les deux (Batch d'abord, puis Interactif)" << std::endl;
    std::cout << "Votre choix (1, 2 ou 3) : ";

    int choix = 1;
    if (std::cin >> choix) {
        if (choix == 1) {
            lancerModeInteractif();
        } else if (choix == 2) {
            lancerModeBatch();
        } else if (choix == 3) {
            lancerModeBatch();
            lancerModeInteractif();
        } else {
            lancerModeInteractif();
        }
    } else {
        lancerModeInteractif();
    }

    return 0;
}
