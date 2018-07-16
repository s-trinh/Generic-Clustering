#include <iostream>
#include <opencv2/opencv.hpp>
#include <visp3/gui/vpDisplayX.h>

using namespace cv;

namespace {
int max_neighbor_dist = 10;

template<typename T>
struct ClusterTypeInfo_t {
  T element;
  int unique_id;

  ClusterTypeInfo_t(const T &e, const int id) : element(e), unique_id(id) { }
};

template<typename T>
std::vector<ClusterTypeInfo_t<T> > findNeighborsPoint(const ClusterTypeInfo_t<T> &current, const std::vector<ClusterTypeInfo_t<T> > &input) {
  std::vector<ClusterTypeInfo_t<T> > neighbors;
  vpImagePoint curPt = current.element;

  for (typename std::vector<ClusterTypeInfo_t<T> >::const_iterator it = input.begin(); it != input.end(); ++it) {
    if (current.unique_id == it->unique_id) {
      continue;
    }

    vpImagePoint imPt = it->element;
    if (vpImagePoint::distance(curPt, imPt) <= max_neighbor_dist) {
      neighbors.push_back(*it);
    }
  }

  return neighbors;
}

template<typename T>
std::map<int, std::vector<T> > clustering(const std::vector<T> &P_,
                                          std::vector<ClusterTypeInfo_t<T> > (*findNeighbors)(const ClusterTypeInfo_t<T> &,
                                                                                              const std::vector<ClusterTypeInfo_t<T> > &),
                                          const int minNeighbors=1) {
    std::vector<ClusterTypeInfo_t<T> > P;
    P.reserve(P_.size());
    for (size_t i = 0; i < P_.size(); i++) {
      P.push_back(ClusterTypeInfo_t<T>(P_[i], i));
    }

    std::map<int, std::vector<T> > clusters;
    std::vector<ClusterTypeInfo_t<T> > Q;

    int idx = 0;
    std::vector<int> already_processed;
    for (size_t i = 0; i < P.size(); i++) {
      if (std::find(already_processed.begin(), already_processed.end(),
          P[i].unique_id) != already_processed.end()) {
        continue;
      }

      Q.push_back(P[i]);
      already_processed.push_back(P[i].unique_id);

      for (size_t j = 0; j < Q.size(); j++) {
        std::vector<ClusterTypeInfo_t<T> > neighbors = findNeighbors(Q[j], P);

        for (size_t k = 0; k < neighbors.size(); k++) {
          if (std::find(already_processed.begin(), already_processed.end(),
              neighbors[k].unique_id) == already_processed.end()) {
            Q.push_back(neighbors[k]);
            already_processed.push_back(neighbors[k].unique_id);
          }
        }
      }

      if (Q.size() >= minNeighbors) {
        for (typename std::vector<ClusterTypeInfo_t<T> >::iterator it = Q.begin(); it != Q.end(); ++it) {
          clusters[idx].push_back(it->element);
        }
      }

      idx++;
      Q.clear();
    }

    return clusters;
}

}

int main(int /*argc*/, char */*argv*/[]) {
  //Let the user initializes some cluster points
  vpImage<unsigned char> I(480, 640);
  bool quit = false;
  vpDisplayX d;
  d.init(I, 0, 0, "Initialize some cluster points");
  vpDisplay::display(I);

  std::vector<vpImagePoint> points;
  while (!quit) {
    vpDisplay::displayText(I, 20, 20, "Left click: add a point, right click: stop adding point", vpColor::red);
    vpDisplay::flush(I);

    vpMouseButton::vpMouseButtonType button;
    vpImagePoint imPt;
    if (vpDisplay::getClick(I, imPt, button, false)) {
      switch (button) {
        case vpMouseButton::button1:
          vpDisplay::displayCross(I, imPt, 8, vpColor::red);
          points.push_back(imPt);
          break;

        case vpMouseButton::button3:
          quit = true;
          break;

        default:
          break;
      }
    }

    vpTime::wait(30);
  }

  //Perform an Euclidean clustering
  const std::string window_name = "Clustering";
  namedWindow(window_name);

  createTrackbar("max neighbors dist", window_name, &max_neighbor_dist, 150, 0);

  while (true) {
    Mat img = Mat::zeros(480, 640, CV_8UC3);

    const int minNeighbors = 0;
    std::map<int, std::vector<vpImagePoint> > clusters = clustering<vpImagePoint>(points, findNeighborsPoint, minNeighbors);
    std::cout << "clusters: " << clusters.size() << std::endl;
    for (std::map<int, std::vector<vpImagePoint> >::const_iterator it1 = clusters.begin(); it1 != clusters.end(); ++it1) {
      std::cout << "clusters " << it1->first << " ; nb points: " << it1->second.size() << std::endl;
    }

    for (std::map<int, std::vector<vpImagePoint> >::const_iterator it1 = clusters.begin(); it1 != clusters.end(); ++it1) {
      vpColor color = vpColor::getColor(it1->first);

      for (std::vector<vpImagePoint>::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
        circle(img, Point(it2->get_u(), it2->get_v()), 8, Scalar(color.B, color.G, color.R), 2);
      }
    }

    imshow(window_name, img);
    if (waitKey(30) == 27)
      break;
  }

  return EXIT_SUCCESS;
}
