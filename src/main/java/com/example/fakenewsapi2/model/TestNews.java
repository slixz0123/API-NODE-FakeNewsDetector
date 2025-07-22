package com.example.fakenewsapi2.model;

import jakarta.persistence.Column;
import jakarta.persistence.Lob;

public class TestNews {
    @Lob
    @Column(name = "text", columnDefinition = "LONGTEXT")
    private String text;

    @Lob
    @Column(name = "title", columnDefinition = "LONGTEXT")
    private String title;
    public boolean expected;

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

    public boolean isExpected() {
        return expected;
    }

    public void setExpected(boolean expected) {
        this.expected = expected;
    }
}