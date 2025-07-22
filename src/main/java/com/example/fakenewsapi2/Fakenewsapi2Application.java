package com.example.fakenewsapi2;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;

@SpringBootApplication
@EntityScan(basePackages = "com.example.fakenewsapi2.model")
public class Fakenewsapi2Application {

    public static void main(String[] args) {
        SpringApplication.run(com.example.fakenewsapi2.Fakenewsapi2Application.class, args);
    }

}