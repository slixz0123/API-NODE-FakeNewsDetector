package com.example.fakenewsapi2.service;

import com.example.fakenewsapi2.model.Prediction;
import com.example.fakenewsapi2.repository.PredictionRepository;
import com.opencsv.CSVReader;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.FileReader;
import java.util.*;

@Service
public class CsvLoadService {
    private static final int BATCH_SIZE = 100;
    @Autowired
    private PredictionRepository predictionRepository;

    @Autowired
    private PredictionService predictionService;

    /**
     * Lee un CSV, infiere con el microservicio FastAPI y guarda cada registro.
     *
     * @param path         ruta al archivo CSV
     * @param isFakeLabel  etiqueta por defecto si falla la inferencia
     * @return número de registros insertados
     */
    @Transactional
    public int loadCsvToDb(String path, boolean isFakeLabel) {
        int count = 0;
        List<Prediction> batch = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new FileReader(path))) {
            String[] headers = reader.readNext();
            if (headers == null) return 0;

            Map<String, Integer> colIndex = new HashMap<>();
            for (int i = 0; i < headers.length; i++) {
                colIndex.put(headers[i].trim().toLowerCase(), i);
            }

            String[] row;
            while ((row = reader.readNext()) != null) {
                String title = getValue(row, colIndex, "title", "subject");
                String text = getValue(row, colIndex, "text", "body");

                Prediction p = new Prediction();
                p.setTitle(title);
                p.setText(text);
                p.setDate(new Date());

                try {
                    Prediction predicted = predictionService.predictTF(p);
                    p.setFake(predicted.isFake());
                    p.setConfidence(predicted.getConfidence());
                } catch (Exception e) {
                    p.setFake(isFakeLabel);
                    p.setConfidence(0.0);
                }

                batch.add(p);
                count++;

                if (batch.size() >= BATCH_SIZE) {
                    predictionRepository.saveAll(batch);
                    batch.clear();
                }
            }

            if (!batch.isEmpty()) {
                predictionRepository.saveAll(batch);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return count;
    }
    private String getValue(String[] row, Map<String, Integer> colIndex, String... keys) {
        for (String key : keys) {
            if (colIndex.containsKey(key)) {
                return row[colIndex.get(key)];
            }
        }
        return "";
    }
    /** Borra todas las predicciones de la base de datos */
    @Transactional
    public void clearAllPredictions() {
        predictionRepository.deleteAll();
    }
}