#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>
#include <unordered_map>


#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 10;

  const double std_x = std[0];
  const double std_y = std[1];
  const double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (unsigned int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
    weights.push_back(1.0);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate){
  const double std_x = std_pos[0];
  const double std_y = std_pos[1];
  const double std_theta = std_pos[2];

  for (unsigned int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 0.001) {
      const double velocity_delta_t = velocity * delta_t;
      particles[i].x += velocity_delta_t * cos(particles[i].theta);
      particles[i].y += velocity_delta_t * sin(particles[i].theta);
    } else {
      const double velocity_yaw_rate = velocity/yaw_rate;
      const double yaw_rate_delta_t = yaw_rate*delta_t;
      particles[i].x += velocity_yaw_rate * (sin(particles[i].theta + yaw_rate_delta_t) - sin(particles[i].theta));
      particles[i].y += velocity_yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_delta_t));
      particles[i].theta += yaw_rate_delta_t;
    }

    normal_distribution<double> dist_x(particles[i].x, std_x);
    normal_distribution<double> dist_y(particles[i].y, std_y);
    normal_distribution<double> dist_theta(particles[i].theta, std_theta);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // For every observation, we want to set the closest predicted landmark id
  const unsigned int observations_size = observations.size();
  const unsigned int predicted_size = predicted.size();

  for (unsigned int i = 0; i < observations_size; i++) {
    LandmarkObs &observation = observations[i];
    double minimum_distance = numeric_limits<double>::max();

    for (unsigned int j = 0; j < predicted_size; j++) {
      const LandmarkObs prediction = predicted[j];
      const double distance = dist(observation.x, observation.y, prediction.x, prediction.y);

      if (minimum_distance > distance) {
        minimum_distance = distance;
        observation.id = prediction.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  const double sigma_x = std_landmark[0];
  const double sigma_y = std_landmark[1];

  for (unsigned int i = 0; i < num_particles; i++) {
    Particle &particle = particles[i];

    const vector<LandmarkObs> map_landmarks_in_range = retrieveMapLandmarksInRange(sensor_range, particle, map_landmarks);

    const vector<LandmarkObs> transformed_observations = transformLocalToMapCoordinates(particle, observations);

    const vector<LandmarkObs> closest_landmarks = findClosestMapLandmarks(map_landmarks_in_range, transformed_observations);

    double probability = calculateParticleWeightBasedOnDistanceBetween(transformed_observations, closest_landmarks, sigma_x, sigma_y);

    particle.weight = probability;
    weights.push_back(probability);
  }
}

std::vector<LandmarkObs> ParticleFilter::retrieveMapLandmarksInRange(const double sensor_range,
                                                                     const Particle particle,
                                                                     const Map map_landmarks) {
  vector<LandmarkObs> map_landmarks_in_range;
  const unsigned int map_landmarks_size = map_landmarks.landmark_list.size();

  for (unsigned int i = 0; i < map_landmarks_size; i++) {
   const Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
   const double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);

   if (distance < sensor_range) {
     LandmarkObs landmark_observed;
     landmark_observed.id = landmark.id_i;
     landmark_observed.x = landmark.x_f;
     landmark_observed.y = landmark.y_f;

     map_landmarks_in_range.push_back(landmark_observed);
   }
  }

  return map_landmarks_in_range;
}

std::vector<LandmarkObs> ParticleFilter::transformLocalToMapCoordinates(const Particle particle,
                                                                        const std::vector<LandmarkObs> observations) {
  const double cos_particle_theta = cos(particle.theta);
  const double sin_particle_theta = sin(particle.theta);
  const unsigned int observations_size = observations.size();

  vector<LandmarkObs> transformed_observations;

  for (unsigned int i = 0; i < observations_size; i++) {
    const LandmarkObs observation = observations[i];

    LandmarkObs transformed_observation;
    transformed_observation.x = particle.x + observation.x * cos_particle_theta - observation.y * sin_particle_theta;
    transformed_observation.y = particle.y + observation.x * sin_particle_theta + observation.y * cos_particle_theta;
    transformed_observation.id = observation.id;

    transformed_observations.push_back(transformed_observation);
  }

  return transformed_observations;
}

std::vector<LandmarkObs> ParticleFilter::findClosestMapLandmarks(const std::vector<LandmarkObs> map_landmarks,
                                                                 const std::vector<LandmarkObs> observations) {
  const unsigned int observations_size = observations.size();
  const unsigned int map_landmark_size = map_landmarks.size();

  std::vector<LandmarkObs> closest_map_landmarks;

  for (unsigned int i = 0; i < observations_size; i++) {
    const LandmarkObs observation = observations[i];
    LandmarkObs closest_map_landmark;

    double minimum_distance = numeric_limits<double>::max();

    for (unsigned int j = 0; j < map_landmark_size; j++) {
      const LandmarkObs map_landmark = map_landmarks[j];
      const double distance = dist(observation.x, observation.y, map_landmark.x, map_landmark.y);

      if (minimum_distance > distance) {
        minimum_distance = distance;
        closest_map_landmark = map_landmark;
      }
    }

    closest_map_landmarks.push_back(closest_map_landmark);
  }

  return closest_map_landmarks;
}

double ParticleFilter::calculateParticleWeightBasedOnDistanceBetween(const std::vector<LandmarkObs> observations,
                                                                     const std::vector<LandmarkObs> map_landmarks,
                                                                     const double sigma_x,
                                                                     const double sigma_y) {
  const double gauss_norm = 1.0/(2 * M_PI * sigma_x * sigma_y);
  const double two_sigma_x_at_two = 2 * sigma_x * sigma_x;
  const double two_sigma_y_at_two = 2 * sigma_y * sigma_y;

  double weight = 1.0;

  const unsigned int observations_size = observations.size();

  for (unsigned int i = 0; i < observations_size; i++) {
    const LandmarkObs observation = observations[i];
    const LandmarkObs map_landmark = map_landmarks[i];

    const double diff_x = observation.x - map_landmark.x;
    const double diff_y = observation.y - map_landmark.y;

    const double exponent_x = (diff_x * diff_x) / two_sigma_x_at_two;
    const double exponent_y = (diff_y * diff_y) / two_sigma_y_at_two;
    const double exponent = exponent_x + exponent_y;

    weight *= gauss_norm * exp(-exponent);
  }

  return weight;
}

void ParticleFilter::resample() {
  vector<Particle> new_particles;
  std::discrete_distribution<> weighted_distribution(weights.begin(), weights.end());
  unsigned int index = 0;

  for(unsigned int i = 0; i < num_particles; i++){
    index = weighted_distribution(gen);
    new_particles.push_back(particles[index]);
  }

  weights.clear();
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
  vector<double> v = best.sense_x;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
  vector<double> v = best.sense_y;
  stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
