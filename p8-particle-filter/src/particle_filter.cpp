/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 10;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {
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

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 0.001) {
      double velocity_delta_t = velocity * delta_t;
      particles[i].x += velocity_delta_t * cos(particles[i].theta);
      particles[i].y += velocity_delta_t * sin(particles[i].theta);
    } else {
      double velocity_yaw_rate = velocity/yaw_rate;
      double yaw_rate_delta_t = yaw_rate*delta_t;
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
  int observations_size = observations.size();
  int predicted_size = predicted.size();

  for (int i = 0; i < observations_size; i++) {
    LandmarkObs &observation = observations[i];
    double minimum_distance = numeric_limits<double>::max();

    for (int j = 0; j < predicted_size; j++) {
      LandmarkObs prediction = predicted[i];

      double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
      if (minimum_distance > distance) {
        minimum_distance = distance;
        observation.id = prediction.id;
      }
    }
  }
}

std::vector<LandmarkObs> ParticleFilter::findClosestMapLandmarks(std::vector<LandmarkObs> map_landmarks, std::vector<LandmarkObs> observations) {
  // For every observation, we want to find the closest map landmark
  std::vector<LandmarkObs> closest_map_landmarks;

  int observations_size = observations.size();
  int map_landmark_size = map_landmarks.size();

  for (int i = 0; i < observations_size; i++) {
    LandmarkObs observation = observations[i];
    LandmarkObs closest_map_landmark;

    double minimum_distance = numeric_limits<double>::max();

    for (int j = 0; j < map_landmark_size; j++) {
      LandmarkObs map_landmark = map_landmarks[j];

      double distance = dist(observation.x, observation.y, map_landmark.x, map_landmark.y);
      if (minimum_distance > distance) {
        minimum_distance = distance;
        closest_map_landmark = map_landmark;
      }
    }

    closest_map_landmarks.push_back(closest_map_landmark);
  }

  return closest_map_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  int observations_size = observations.size();
  int map_landmarks_size = map_landmarks.landmark_list.size();

  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];
  double gauss_norm = 1.0/(2 * M_PI * sigma_x * sigma_y);

  for (int i = 0; i < num_particles; i++) {
    Particle &particle = particles[i];

    // transform observation coordinates from vehicle coordinates to map coordinates

    double cos_particle_theta = cos(particle.theta);
    double sin_particle_theta = sin(particle.theta);

    vector<LandmarkObs> transformed_observations;
    for (int j = 0; j < observations_size; j++) {
      LandmarkObs observation = observations[j];

      LandmarkObs transformed_observation;
      transformed_observation.x = particle.x + observation.x * cos_particle_theta - observation.y * sin_particle_theta;
      transformed_observation.y = particle.y + observation.x * sin_particle_theta + observation.y * cos_particle_theta;
      transformed_observation.id = observation.id;

      transformed_observations.push_back(transformed_observation);
    }

    // gather all map landmarks that are in the sensor_range of the particle

    vector<LandmarkObs> map_landmarks_in_range;
    for (int j = 0; j < map_landmarks_size; j++) {
      Map::single_landmark_s landmark = map_landmarks.landmark_list[j];
      double distance = dist(landmark.x_f, landmark.y_f, particle.x, particle.y);

      if (distance < sensor_range) {
        LandmarkObs landmark_observed;
        landmark_observed.id = landmark.id_i;
        landmark_observed.x = landmark.x_f;
        landmark_observed.y = landmark.y_f;

        map_landmarks_in_range.push_back(landmark_observed);
      }
    }

    vector<LandmarkObs> closest_landmarks = findClosestMapLandmarks(map_landmarks_in_range, transformed_observations);

    // for all landmarks in range, we calculate the position difference between
    // the predicted observation and the real landmark

    double probability = 1.0;

    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    for (int j = 0; j < observations_size; j++) {
      LandmarkObs observation = transformed_observations[j];
      LandmarkObs map_landmark = closest_landmarks[j];

      // weights

      double diff_x = observation.x - map_landmark.x;
      double diff_y = observation.y - map_landmark.y;

      double exponent_x = (diff_x * diff_x) / (2 * sigma_x * sigma_x);
      double exponent_y = (diff_y * diff_y) / (2 * sigma_y * sigma_y);
      double exponent = exponent_x + exponent_y;

      double weight = gauss_norm * exp(-exponent_x) * exp(-exponent_y);
      probability *= weight;

      associations.push_back(observation.id);
      sense_x.push_back(observation.x);
      sense_y.push_back(observation.y);
    }

    particle.weight = probability;
    weights[i] = probability;

    SetAssociations(particle, associations, sense_x, sense_y);

    cout << weights[i] << endl;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  std::vector<Particle> new_particles(num_particles);
  std::discrete_distribution<> weighted_distribution(weights.begin(), weights.end());
  int index = 0;

  for (int i = 0; i < num_particles; i++) {
    index = weighted_distribution(gen);
    new_particles.push_back(particles[index]);
  }

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
