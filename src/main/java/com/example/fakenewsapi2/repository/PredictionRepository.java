package com.example.fakenewsapi2.repository;

import com.example.fakenewsapi2.model.Prediction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    List<Prediction> findTop10ByOrderByDateDesc();
    List<Prediction> findByIsFakeTrueOrderByDateDesc();
    List<Prediction> findByIsFakeFalseOrderByDateDesc();
}

