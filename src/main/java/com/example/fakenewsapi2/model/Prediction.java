package com.example.fakenewsapi2.model;

import jakarta.persistence.*;

import java.util.Date;

@Entity
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)  // Asegura que el ID se genere automáticamente
    private Long id;

    @Lob
    @Column(name = "text", columnDefinition = "LONGTEXT")
    private String text;

    @Lob
    @Column(name = "title", columnDefinition = "LONGTEXT")
    private String title;
    private boolean isFake;
    private double confidence;
    private Date date;

    public Date getDate() {
        return date;
    }

    public void setDate(Date date) {
        this.date = date;
    }

    // Constructor vacío
    public Prediction() {
    }

    // Getters y Setters

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public boolean isFake() {
        return isFake;
    }

    public void setFake(boolean fake) {
        isFake = fake;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }
}
