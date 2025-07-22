package com.example.fakenewsapi2.controller;

import com.example.fakenewsapi2.model.Prediction;
import com.example.fakenewsapi2.repository.PredictionRepository;
import com.example.fakenewsapi2.service.CsvLoadService;
import com.example.fakenewsapi2.service.PredictionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Date;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/predict")
public class PredictionController {

    private final CsvLoadService csvLoadService;
    private final PredictionService predictionService;
    private final PredictionRepository predictionRepository;

    @Autowired
    public PredictionController(CsvLoadService csvLoadService,
                                PredictionService predictionService,
                                PredictionRepository predictionRepository) {
        this.csvLoadService = csvLoadService;
        this.predictionService = predictionService;
        this.predictionRepository = predictionRepository;
    }

    @PostMapping("/simple")
    public Prediction predictSimple(@RequestBody Prediction incoming) {
        Prediction result = predictionService.predictTF(incoming);
        result.setDate(new Date());
        return predictionRepository.save(result);
    }

    @PostMapping
    public ResponseEntity<Prediction> predict(
            @RequestBody Prediction incoming,
            @RequestParam(required = false, defaultValue = "tf") String model) {

        incoming.setDate(new Date());
        Prediction result;

        try {
            switch (model.toLowerCase()) {
                case "tf":
                    result = predictionService.predictTF(incoming);
                    break;
                case "lr":
                    result = predictionService.predictLR(incoming);
                    break;
                case "xgb":
                    result = predictionService.predictXGB(incoming);
                    break;
                default:
                    throw new IllegalArgumentException("Modelo no válido: " + model);
            }

            result.setDate(new Date());
            Prediction saved = predictionRepository.save(result);
            return ResponseEntity.ok(saved);

        } catch (Exception ex) {
            incoming.setFake(false);
            incoming.setConfidence(0.0);
            Prediction saved = predictionRepository.save(incoming);
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(saved);
        }
    }

    @PostMapping("/load-all")
    public Map<String, String> loadAll() {
        int total = 0;
        total += csvLoadService.loadCsvToDb("src/main/resources/data/True.csv", false);
        total += csvLoadService.loadCsvToDb("src/main/resources/data/Fake.csv", true);
        total += csvLoadService.loadCsvToDb("src/main/resources/data/onlytrue1000.csv", false);
        total += csvLoadService.loadCsvToDb("src/main/resources/data/onlyfakes1000.csv", true);
        return Map.of("totalImported", String.valueOf(total));
    }

    // CORRECCIÓN AQUÍ - CAMBIO DE retrain() A retrainTF()
    @PostMapping("/retrain")
    public ResponseEntity<String> retrain() {
        try {
            String out = predictionService.retrainTF();  // Método corregido
            return ResponseEntity.ok("Retrain OK:\n" + out);
        } catch (Exception ex) {
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Retrain failed: " + ex.getMessage());
        }
    }

    @GetMapping("/recent")
    public List<Prediction> getRecent() {
        return predictionRepository.findTop10ByOrderByDateDesc();
    }

    @GetMapping("/fake")
    public List<Prediction> getFake() {
        return predictionRepository.findByIsFakeTrueOrderByDateDesc();
    }

    @GetMapping("/true")
    public List<Prediction> getTrue() {
        return predictionRepository.findByIsFakeFalseOrderByDateDesc();
    }

    @GetMapping("/all")
    public List<Prediction> getAll() {
        return predictionRepository.findAll();
    }

    @PostMapping("/ensemble")
    public Prediction ensemblePredict(@RequestBody Prediction incoming) {
        return predictionService.ensemblePredict(incoming);
    }
}