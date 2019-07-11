package groupP;

import apgas.Configuration;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ThreadLocalRandom;

import static apgas.Constructs.*;

public class BuddyLcmSeq {
    public static void main(String[] args) {

        // Arraylänge n × n
        final int n = Integer.valueOf(args[0]);

        // Obergrenze für Zufallszahlen (exklusive m)
        final int m = Integer.valueOf(args[1]);

        // Initialwert des Zufallszahlengenerators für Array a und b
        final int seedA = Integer.valueOf(args[2]);
        final int seedB = Integer.valueOf(args[3]);

        // Schwellwert für zwei Buddies
        final int minLcm = Integer.valueOf(args[4]);

        // Wenn true, dann werden zusätzlich noch a, b, alle gefundenen Buddy Positionen in b
        // und die zugehörigen lcms ausgegeben
        final boolean verbose = Boolean.valueOf(args[5]);

        final int t = numLocalWorkers();

        if (n % t * places().size() != 0 || n < t * places().size()) {
            System.out.println("n must be divisable by t * p!");
            System.exit(1);
        }

        final Integer[][] a = new Integer[n][n];
        final Integer[][] b = new Integer[n][n];

        final HashMap<Integer, Buddy> aMap = new HashMap<>();
        final TreeMap<Integer, Coordinates> bMap = new TreeMap<>();

        Random random = new Random();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                random.setSeed(seedA + Integer.parseInt(i + "" + j));
                a[i][j] = random.nextInt(m) + 1;
                aMap.put(a[i][j], null);
                random.setSeed(seedB + Integer.parseInt(i + "" + j));
                b[i][j] = random.nextInt(m) + 1;
                bMap.put(b[i][j], new Coordinates(i, j));
            }
        }

        final long start = System.nanoTime();

        aMap.forEach((aKey, aBuddy) -> {
            int largestLcmFound = -1;
            Coordinates largestLcmFoundCoordinates = null;

            for (Map.Entry<Integer, Coordinates> bEntry : bMap.entrySet()) {
                final int lcm = lcm(aKey, bEntry.getKey(), gcd(aKey, bEntry.getKey()));

                if (lcm >= minLcm) {
                    if (minLcm == -1) {
                        if (lcm > largestLcmFound) {
                            largestLcmFound = lcm;
                            largestLcmFoundCoordinates = bEntry.getValue();
                        }
                    } else {
                        aMap.put(aKey, new Buddy(bEntry.getValue(), lcm));
                        break;
                    }
                }
            }

            if (minLcm == -1 && largestLcmFoundCoordinates != null) {
                aMap.put(aKey, new Buddy(largestLcmFoundCoordinates, largestLcmFound));
            }
        });

       final StringBuilder aStringBuilder = new StringBuilder();
       final StringBuilder bStringBuilder = new StringBuilder();
       final StringBuilder noBuddiesStringBuilder = new StringBuilder("[");
       final StringBuilder buddyPositionsStringBuilder = new StringBuilder();
       final StringBuilder lcmsStringBuilder = new StringBuilder();

       boolean oneNoBuddyFound = false;

       for (int i = 0; i < n; i++) {
           aStringBuilder.append("[");
           bStringBuilder.append("[");
           buddyPositionsStringBuilder.append("[");
           lcmsStringBuilder.append("[");

           for (int j = 0; j < n; j++) {
               Buddy buddy = aMap.get(a[i][j]);

               if (buddy == null) {
                   if (oneNoBuddyFound) {
                       noBuddiesStringBuilder.append(", ");
                   }

                   noBuddiesStringBuilder.append(i).append(":").append(j);

                   oneNoBuddyFound = true;
               }

               if (j != 0) {
                   aStringBuilder.append(", ");
                   bStringBuilder.append(", ");
                   buddyPositionsStringBuilder.append(", ");
                   lcmsStringBuilder.append(", ");
               }

               aStringBuilder.append(a[i][j]);
               bStringBuilder.append(b[i][j]);
               buddyPositionsStringBuilder.append(buddy != null ? buddy.bCoordinates : "-1:-1");
               lcmsStringBuilder.append(buddy != null ? buddy.lcm : "-1");
           }

           aStringBuilder.append("]\n");
           bStringBuilder.append("]\n");
           buddyPositionsStringBuilder.append("]\n");
           lcmsStringBuilder.append("]\n");
       }

       noBuddiesStringBuilder.append("]");

       if (noBuddiesStringBuilder.toString().equals("[]")) {
           System.out.println("all elements have found buddies");
       } else {
           System.out.println("there are no buddies for the following elements");
           System.out.println(noBuddiesStringBuilder);
       }

       if (verbose) {
           System.out.print("a =\n" + aStringBuilder);
           System.out.print("b =\n" + bStringBuilder);

           System.out.print("Found buddy positions in b=\n" +  buddyPositionsStringBuilder);
           System.out.print("Found kgVs=\n" +  lcmsStringBuilder);
       }

        System.out.println("Process time = " + ((System.nanoTime() - start) / 1E9D) + " sec");
    }

    // https://de.wikipedia.org/wiki/Gr%C3%B6%C3%9Fter_gemeinsamer_Teiler#Euklidischer_und_steinscher_Algorithmus
    // without usage of "abs" as only positive random numbers are checked
    private static int gcd(int a, int b) {
        int h;

        if (a == 0) return b;
        if (b == 0) return a;

        do {
            h = a % b;
            a = b;
            b = h;
        } while (b != 0);

        return a;
    }

    // https://de.wikipedia.org/wiki/Kleinstes_gemeinsames_Vielfaches#Berechnung_%C3%BCber_den_gr%C3%B6%C3%9Ften_gemeinsamen_Teiler_(ggT)
    private static int lcm(final int a, final int b, final int gcd) {
        return (a / gcd) * b;
    }

    private static String matrixToString(final Object[][] m) {
        return Arrays.deepToString(m)
                .replaceAll("], ", "]\n")
                .replace("[[", "[")
                .replace("]]", "]");
    }

    static class Coordinates {
        private int x;
        private int y;

        Coordinates(final int x, final int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            return x + ":" + y;
        }
    }

    static class Buddy {
        private Coordinates bCoordinates;
        private int lcm;

        Buddy(final Coordinates bCoordinates, final int lcm) {
            this.bCoordinates = bCoordinates;
            this.lcm = lcm;
        }
    }
}
