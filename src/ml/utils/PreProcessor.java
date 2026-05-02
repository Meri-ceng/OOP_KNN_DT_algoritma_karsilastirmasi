package ml.utils;

import ml.model.UserRecord;
import java.util.*;

/**
 * Ham veriyi makine öğrenmesi algoritmalarına hazırlayan ön işleme sınıfı.
 * - Label Encoding: Kategorik verileri sayısal ID'lere çevirir.
 * - Min-Max Normalizasyon: Sayısal değerleri [0-1] aralığına çeker.
 */
public class PreProcessor {

    // Encoding haritaları
    private final Map<String, Double> genderMap    = new LinkedHashMap<>();
    private final Map<String, Double> brandCodeMap = new LinkedHashMap<>();

    // Normalizasyon için min-max değerleri (eğitim verisinden öğrenilir)
    private double minLineNet, maxLineNet;
    private double minBrandCode, maxBrandCode;

    private boolean fitted = false;

    /**
     * Encoding ve normalizasyon parametrelerini eğitim verisi üzerinden öğrenir.
     * @param trainingData Eğitim kayıtları
     */
    public void fit(List<UserRecord> trainingData) {
        // --- Gender Encoding ---
        genderMap.clear();
        genderMap.put("Female", 0.0);
        genderMap.put("female", 0.0);
        genderMap.put("F",      0.0);
        genderMap.put("Male",   1.0);
        genderMap.put("male",   1.0);
        genderMap.put("M",      1.0);

        // --- BrandCode Encoding (unique brand kodları sıralı sayısal ID'e) ---
        brandCodeMap.clear();
        Set<String> brandCodes = new TreeSet<>();
        for (UserRecord r : trainingData) brandCodes.add(r.getBrandCode());
        double idx = 0;
        for (String bc : brandCodes) brandCodeMap.put(bc, idx++);

        // --- Min-Max için sınır değerleri ---
        minLineNet   = Double.MAX_VALUE; maxLineNet   = Double.MIN_VALUE;
        minBrandCode = 0;               maxBrandCode = Math.max(1, brandCodeMap.size() - 1);

        for (UserRecord r : trainingData) {
            double v = r.getLineNetTotal();
            if (v < minLineNet)   minLineNet   = v;
            if (v > maxLineNet)   maxLineNet   = v;
        }
        if (maxLineNet == minLineNet) maxLineNet = minLineNet + 1; // sıfıra bölme koruması

        fitted = true;
    }

    /**
     * Verilen kayıt listesini dönüştürür (fit sonrası çağrılmalıdır).
     */
    public void transform(List<UserRecord> data) {
        if (!fitted) throw new IllegalStateException("PreProcessor.fit() önce çağrılmalıdır.");
        for (UserRecord r : data) {
            // Gender encoding
            r.setEncodedGender(genderMap.getOrDefault(r.getGender(), 0.0));

            // BrandCode encoding
            double bcEncoded = brandCodeMap.getOrDefault(r.getBrandCode(), 0.0);
            r.setEncodedBrandCode(bcEncoded);

            // Min-Max normalizasyon
            r.setNormalizedLineNetTotal(minMax(r.getLineNetTotal(), minLineNet, maxLineNet));
        }
    }

    /** fit + transform aynı veri üzerinde */
    public void fitTransform(List<UserRecord> data) {
        fit(data);
        transform(data);
    }

    private double minMax(double value, double min, double max) {
        return (value - min) / (max - min);
    }

    // Bilgi metodları
    public Map<String, Double> getGenderMap()    { return Collections.unmodifiableMap(genderMap); }
    public Map<String, Double> getBrandCodeMap() { return Collections.unmodifiableMap(brandCodeMap); }
    public int getBrandCount()                   { return brandCodeMap.size(); }
}
