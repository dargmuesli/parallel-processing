package groupP;

import apgas.Place;
import apgas.util.GlobalRef;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.function.BiConsumer;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import static apgas.Constructs.*;

public class BuddyLcm {
    public static void main(String[] args) {
        // validate argument length
        if (args.length != 6) {
            throw new IllegalArgumentException("Invalid argument count");
        }

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
        final int p = places().size();

        if (n % t * places().size() != 0 || n < t * places().size()) {
            throw new IllegalArgumentException("n must be divisable by t * p!");
        }

        final long start = System.nanoTime();

        GlobalRef<int[][]> aPlace = new GlobalRef<>(places(), () -> new int[n / p][n]);
        GlobalRef<int[][]> bPlace = new GlobalRef<>(places(), () -> new int[n / p][n]);

        GlobalRef<ConcurrentHashMap<Integer, Buddy>> aPlaceBuddyMap = new GlobalRef<>(places(), ConcurrentHashMap::new);
        GlobalRef<List<Map<Integer, Buddy>>> aPlaceBuddyMapList = new GlobalRef<>(places(), ArrayList::new);
        GlobalRef<ConcurrentSkipListMap<Integer, Coordinates>> bPlaceCoordinatesMap = new GlobalRef<>(places(), ConcurrentSkipListMap::new);

        GlobalRef<TreeMap<Integer, String>> noBuddiesOutMap = new GlobalRef<>(places(), TreeMap::new),
                aOutMap = new GlobalRef<>(places(), TreeMap::new),
                bOutMap = new GlobalRef<>(places(), TreeMap::new),
                buddyPositionsOutMap = new GlobalRef<>(places(), TreeMap::new),
                lcmsOutMap = new GlobalRef<>(places(), TreeMap::new);

        finish(() -> {
            for (final Place place : places()) {
                asyncAt(place, () -> {
                    finish(() -> {
                        for (int worker = 0; worker < t; worker++) {
                            final int fWorker = worker;

                            async(() -> {
                                final Random random = new Random();

                                for (int row = 0; row < n / (p * t); row++) {
                                    for (int column = 0; column < n; column++) {
                                        int localTotalRow = fWorker * (n / (p * t)) + row;
                                        int totalRow = here().id * t + localTotalRow;

                                        random.setSeed(seedA + Integer.parseInt(totalRow + "" + column));
                                        int aValue = random.nextInt(m) + 1;
                                        aPlace.get()[localTotalRow][column] = aValue;
                                        aPlaceBuddyMap.get().put(aValue, new Buddy(null, null));

                                        random.setSeed(seedB + Integer.parseInt(totalRow + "" + column));
                                        int bValue = random.nextInt(m) + 1;
                                        bPlace.get()[localTotalRow][column] = bValue;
                                        bPlaceCoordinatesMap.get().put(bValue, new Coordinates(totalRow, column));
                                    }
                                }
                            });
                        }
                    });

                    final Collector<Map.Entry<Integer, Buddy>, ?, List<Map<Integer, Buddy>>> batchesCollector = unorderedBatches(aPlaceBuddyMap.get().size() / t, Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue), Collectors.toList());
                    aPlaceBuddyMapList.set(aPlaceBuddyMap.get().entrySet().stream().collect(batchesCollector));

                    finish(() -> {
                        for (int worker = 0; worker < t; worker++) {
                            final int fWorker = worker;

                            async(() -> aPlaceBuddyMapList.get().get(fWorker).forEach((aKey, aBuddy) -> {
                                int largestLcmFound = -1;
                                Coordinates largestLcmFoundCoordinates = null;

                                for (Map.Entry<Integer, Coordinates> bEntry : bPlaceCoordinatesMap.get().entrySet()) {
                                    final int lcm = lcm(aKey, bEntry.getKey(), gcd(aKey, bEntry.getKey()));

                                    if (lcm >= minLcm) {
                                        if (minLcm == -1) {
                                            if (lcm > largestLcmFound) {
                                                largestLcmFound = lcm;
                                                largestLcmFoundCoordinates = bEntry.getValue();
                                            }
                                        } else {
                                            aPlaceBuddyMap.get().put(aKey, new Buddy(bEntry.getValue(), lcm));
                                            break;
                                        }
                                    }
                                }

                                if (minLcm == -1 && largestLcmFoundCoordinates != null) {
                                    aPlaceBuddyMap.get().put(aKey, new Buddy(largestLcmFoundCoordinates, largestLcmFound));
                                }
                            }));
                        }
                    });

                    final StringBuilder aStringBuilder = new StringBuilder(),
                            bStringBuilder = new StringBuilder(),
                            noBuddiesStringBuilder = new StringBuilder(),
                            buddyPositionsStringBuilder = new StringBuilder(),
                            lcmsStringBuilder = new StringBuilder();

                    boolean oneNoBuddyFound = false;

                    for (int row = 0; row < n / p; row++) {
                        if (verbose) {
                            aStringBuilder.append("[");
                            bStringBuilder.append("[");
                            buddyPositionsStringBuilder.append("[");
                            lcmsStringBuilder.append("[");
                        }

                        for (int column = 0; column < n; column++) {
                            Buddy buddy = aPlaceBuddyMap.get().get(aPlace.get()[row][column]);

                            if (buddy.lcm == null) {
                                if (oneNoBuddyFound) {
                                    noBuddiesStringBuilder.append(", ");
                                }

                                noBuddiesStringBuilder.append(here().id * (n / p) + row).append(":").append(column);

                                oneNoBuddyFound = true;
                            }

                            if (verbose) {
                                if (column != 0) {
                                    aStringBuilder.append(", ");
                                    bStringBuilder.append(", ");
                                    buddyPositionsStringBuilder.append(", ");
                                    lcmsStringBuilder.append(", ");
                                }

                                aStringBuilder.append(aPlace.get()[row][column]);
                                bStringBuilder.append(bPlace.get()[row][column]);
                                buddyPositionsStringBuilder.append(buddy.lcm != null ? buddy.bCoordinates : "-1:-1");
                                lcmsStringBuilder.append(buddy.lcm != null ? buddy.lcm : "-1");
                            }
                        }

                        if (verbose) {
                            aStringBuilder.append("]\n");
                            bStringBuilder.append("]\n");
                            buddyPositionsStringBuilder.append("]\n");
                            lcmsStringBuilder.append("]\n");
                        }
                    }

                    if (noBuddiesStringBuilder.length() > 0) {
                        asyncAt(noBuddiesOutMap.home(), () -> noBuddiesOutMap.get().put(place.id, noBuddiesStringBuilder.toString()));
                    }

                    if (verbose) {
                        asyncAt(aOutMap.home(), () -> aOutMap.get().put(place.id, aStringBuilder.toString()));
                        asyncAt(bOutMap.home(), () -> bOutMap.get().put(place.id, bStringBuilder.toString()));
                        asyncAt(buddyPositionsOutMap.home(), () -> buddyPositionsOutMap.get().put(place.id, buddyPositionsStringBuilder.toString()));
                        asyncAt(lcmsOutMap.home(), () -> lcmsOutMap.get().put(place.id, lcmsStringBuilder.toString()));
                    }
                });
            }
        });

        if (noBuddiesOutMap.get().size() == 0) {
            System.out.println("all elements have found buddies");
        } else {
            System.out.print("there are no buddies for the following elements\n[");
            noBuddiesOutMap.get().forEach((integer, s) -> System.out.print(s));
            System.out.println("]");
        }

        if (verbose) {
            System.out.println("a =");
            aOutMap.get().forEach((integer, s) -> System.out.print(s));
            System.out.println("b =");
            bOutMap.get().forEach((integer, s) -> System.out.print(s));
            System.out.println("Found buddy positions in b=");
            buddyPositionsOutMap.get().forEach((integer, s) -> System.out.print(s));
            System.out.println("Found lcms=");
            lcmsOutMap.get().forEach((integer, s) -> System.out.print(s));
        }

        System.out.println("Process time = " + ((System.nanoTime() - start) / 1E9D) + " sec");
    }

    // https://de.wikipedia.org/wiki/Euklidischer_Algorithmus#Beschreibung_durch_Pseudocode
    private static int gcd(int a, int b) {
        if (a == 0) {
            return b;
        } else {
            while (b != 0) {
                if (a > b) {
                    a -= b;
                } else {
                    b -= a;
                }
            }

            return a;
        }
    }

    // https://de.wikipedia.org/wiki/Gr%C3%B6%C3%9Fter_gemeinsamer_Teiler#Euklidischer_und_steinscher_Algorithmus
    // without usage of "abs" as only positive random numbers are checked
    private static int gcd_modern(int a, int b) {
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
        private Integer lcm;

        Buddy(final Coordinates bCoordinates, final Integer lcm) {
            this.bCoordinates = bCoordinates;
            this.lcm = lcm;
        }
    }

    // source: https://stackoverflow.com/a/32435407/4682621
    private static <T, A, R> Collector<T, ?, R> unorderedBatches(int batchSize, Collector<List<T>, A, R> downstream) {
        class Acc {
            private List<T> cur = new ArrayList<>();
            private A acc = downstream.supplier().get();
        }

        BiConsumer<Acc, T> accumulator = (acc, t) -> {
            acc.cur.add(t);

            if (acc.cur.size() == batchSize) {
                downstream.accumulator().accept(acc.acc, acc.cur);
                acc.cur = new ArrayList<>();
            }
        };

        return Collector.of(Acc::new, accumulator,
                (acc1, acc2) -> {
                    acc1.acc = downstream.combiner().apply(acc1.acc, acc2.acc);
                    for (T t : acc2.cur) accumulator.accept(acc1, t);
                    return acc1;
                }, acc -> {
                    if (!acc.cur.isEmpty())
                        downstream.accumulator().accept(acc.acc, acc.cur);
                    return downstream.finisher().apply(acc.acc);
                }, Collector.Characteristics.UNORDERED);
    }

    public static <T, AA, A, B, R> Collector<T, ?, R> unorderedBatches(int batchSize,
                                                                       Collector<T, AA, B> batchCollector,
                                                                       Collector<B, A, R> downstream) {
        return unorderedBatches(batchSize,
                Collectors.mapping(list -> list.stream().collect(batchCollector), downstream));
    }
}
