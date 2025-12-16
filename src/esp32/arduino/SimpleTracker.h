// --- FILE: SimpleTracker.h ---
#ifndef SIMPLE_TRACKER_H
#define SIMPLE_TRACKER_H

#include <vector>
#include <map>
#include <cmath>

struct Point {
  int x, y;
};

struct TrackedObject {
  int id;
  Point center;
  std::vector<int> history_y;
  bool counted;
  int disappeared_frames;
};

class SimpleTracker {
public:
  std::map<int, TrackedObject> objects;

  SimpleTracker(int maxDisappeared = 5, int maxDistance = 50) {
    this->maxDisappeared = maxDisappeared;
    this->maxDistance = maxDistance;
  }

  std::map<int, TrackedObject> update(std::vector<Point> inputCentroids) {
    if (inputCentroids.empty()) {
      for (auto it = objects.begin(); it != objects.end(); ) {
        it->second.disappeared_frames++;
        if (it->second.disappeared_frames > maxDisappeared) it = objects.erase(it);
        else ++it;
      }
      return objects;
    }

    if (objects.empty()) {
      for (auto& p : inputCentroids) registerObject(p);
      return objects;
    }

    std::vector<bool> usedInput(inputCentroids.size(), false);
    for (auto& obj : objects) {
      double minDist = maxDistance;
      int minIndex = -1;
      for (int i = 0; i < inputCentroids.size(); i++) {
        if (usedInput[i]) continue;
        double d = calculateDistance(obj.second.center, inputCentroids[i]);
        if (d < minDist) { minDist = d; minIndex = i; }
      }

      if (minIndex != -1) {
        obj.second.center = inputCentroids[minIndex];
        obj.second.history_y.push_back(inputCentroids[minIndex].y);
        if (obj.second.history_y.size() > 5) obj.second.history_y.erase(obj.second.history_y.begin());
        obj.second.disappeared_frames = 0;
        usedInput[minIndex] = true;
      } else {
        obj.second.disappeared_frames++;
      }
    }

    for (auto it = objects.begin(); it != objects.end(); ) {
      if (it->second.disappeared_frames > maxDisappeared) it = objects.erase(it);
      else ++it;
    }

    for (int i = 0; i < inputCentroids.size(); i++) {
      if (!usedInput[i]) registerObject(inputCentroids[i]);
    }
      return objects;
  }

  void registerObject(Point c) {
    TrackedObject obj;
    obj.id = nextObjectID++;
    obj.center = c;
    obj.history_y.push_back(c.y);
    obj.counted = false;
    obj.disappeared_frames = 0;
    objects[obj.id] = obj;
  }
private:
  int nextObjectID = 0;
  int maxDisappeared;
  int maxDistance;

  double calculateDistance(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
  }
};
#endif