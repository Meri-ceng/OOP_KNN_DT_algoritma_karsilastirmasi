package ml.utils;

import ml.model.UserRecord;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.util.*;

/**
 * .xlsx dosyasını okuyarak UserRecord nesnelerine dönüştüren yardımcı sınıf.
 * (Forum onayı: Excel okuma için Apache POI kullanımı serbesttir.)
 */
public class DataLoader {

    // Beklenen sütun indeksleri (MarketSalesKocaeli.xlsx yapısına göre)
    private static final int COL_CLIENT_CODE   = 0;
    private static final int COL_GENDER        = 1;
    private static final int COL_LINE_NET_TOTAL= 2;
    private static final int COL_BRAND         = 3;
    private static final int COL_BRAND_CODE    = 4;
    private static final int COL_CATEGORY      = 5;

    private int skippedRows = 0;

    /**
     * Dosyayı okur ve UserRecord listesini döndürür.
     * @param filePath xlsx dosya yolu
     * @return Temizlenmiş kayıt listesi
     * @throws IOException dosya okunamadığında
     */
    public List<UserRecord> load(String filePath) throws IOException {
        List<UserRecord> records = new ArrayList<>();
        skippedRows = 0;

        try (FileInputStream fis = new FileInputStream(filePath);
             Workbook workbook = new XSSFWorkbook(fis)) {

            Sheet sheet = workbook.getSheetAt(0);
            boolean firstRow = true;

            for (Row row : sheet) {
                // Başlık satırını atla
                if (firstRow) { firstRow = false; continue; }

                try {
                    String clientCode  = getCellString(row, COL_CLIENT_CODE);
                    String gender      = getCellString(row, COL_GENDER);
                    double lineNetTotal= getCellDouble(row, COL_LINE_NET_TOTAL);
                    String brand       = getCellString(row, COL_BRAND);
                    String brandCode   = getCellString(row, COL_BRAND_CODE);
                    String category    = getCellString(row, COL_CATEGORY);

                    // Boş alan kontrolü (Veri Temizleme)
                    if (clientCode.isEmpty() || gender.isEmpty() ||
                        brand.isEmpty()      || category.isEmpty()) {
                        skippedRows++;
                        continue;
                    }

                    records.add(new UserRecord(clientCode, gender, lineNetTotal,
                                               brand, brandCode, category));

                } catch (Exception e) {
                    // Hatalı satırları sessizce atla
                    skippedRows++;
                }
            }
        }
        return records;
    }

    public int getSkippedRows() { return skippedRows; }

    // -----------------------------------------------------------------------
    private String getCellString(Row row, int colIdx) {
        Cell cell = row.getCell(colIdx);
        if (cell == null) return "";
        if (cell.getCellType() == CellType.NUMERIC) {
            double val = cell.getNumericCellValue();
            // Tam sayı ise "1234" formatında döndür
            if (val == Math.floor(val)) return String.valueOf((long) val);
            return String.valueOf(val);
        }
        return cell.getStringCellValue().trim();
    }

    private double getCellDouble(Row row, int colIdx) {
        Cell cell = row.getCell(colIdx);
        if (cell == null) return 0.0;
        if (cell.getCellType() == CellType.NUMERIC) return cell.getNumericCellValue();
        // Metin hücresini sayıya çevirmeyi dene
        try { return Double.parseDouble(cell.getStringCellValue().replace(",", ".")); }
        catch (NumberFormatException e) { return 0.0; }
    }
}
