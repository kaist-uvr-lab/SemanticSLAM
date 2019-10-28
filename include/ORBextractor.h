//
// Created by UVR-KAIST on 2019-02-19.
//


#ifndef UVR_SLAM_ORBEXTRACTOR_H
#define UVR_SLAM_ORBEXTRACTOR_H
#pragma once

#include <vector>
#include <list>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

namespace UVR_SLAM {
    class ExtractorNode
    {
    public:
        ExtractorNode():bNoMore(false){}

        void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

        std::vector<cv::KeyPoint> vKeys;
        cv::Point2i UL, UR, BL, BR;
        std::list<ExtractorNode>::iterator lit;
        bool bNoMore;
    };

    class ORBextractor
    {
    public:

        enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

        ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                     int iniThFAST, int minThFAST);

        ~ORBextractor() {}

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        virtual void operator()(cv::InputArray image, cv::InputArray mask,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::OutputArray descriptors);

        virtual void operator()(cv::InputArray image, cv::InputArray mask,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::OutputArray descriptors, std::vector<cv::Rect> vROI, bool bBackground) { }

        virtual void operator()(cv::InputArray image, cv::InputArray mask,
                                std::vector<cv::KeyPoint>& keypoints,
                                cv::OutputArray descriptors, std::vector<cv::Rect> TrackingROI, std::vector<cv::Rect> vRemovingROI, bool bBackground) {}
        void inline SetNumFeatures(int n){
            nfeatures = n;
        }
        int inline GetLevels(){
            return nlevels;}

        float inline GetScaleFactor(){
            return scaleFactor;}

        std::vector<float> inline GetScaleFactors(){
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors(){
            return mvInvScaleFactor;
        }

        std::vector<float> inline GetScaleSigmaSquares(){
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares(){
            return mvInvLevelSigma2;
        }

        std::vector<cv::Mat> mvImagePyramid;

    protected:

        void ComputePyramid(cv::Mat image);
        virtual void ComputePyramid(cv::Mat image, std::vector<cv::Rect> vRect) {}
        void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
        virtual void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, std::vector<cv::Rect> vROI, bool bBackground) {}
        virtual void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, cv::Rect targetROI, std::vector<cv::Rect> vRemovingROI) {}
        //virtual void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints, std::vector<cv::Rect> vRect) {}

        virtual std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                            const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
        std::vector<cv::Point> pattern;

        int nfeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;

    };
}

#endif //ANDROIDOPENCVPLUGINPROJECT_ORBEXTRACTOR_H
