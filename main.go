package main

import (
	"embed"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/MaxHalford/eaopt"
)

func main() {
	if err := run(); err != nil {
		fmt.Printf("fatal error in run(): %v", err)
	}
}

type arrivalSample struct {
	station  string
	minutes  int // Minutes since 7:00am
	arrivals int
}

func getArrivals(rdr io.Reader) ([]arrivalSample, error) {
	records, err := csv.NewReader(rdr).ReadAll()
	if err != nil {
		return nil, err
	}
	samples := make([]arrivalSample, 0, len(records))
	for _, r := range records {
		split := strings.Split(r[1], ":")
		hour, err := strconv.Atoi(split[0])
		if err != nil {
			return nil, err
		}
		mins, err := strconv.Atoi(split[1])
		if err != nil {
			return nil, err
		}
		arrivals, err := strconv.Atoi(r[2])
		if err != nil {
			return nil, err
		}
		samples = append(samples, arrivalSample{
			station:  r[0],
			minutes:  ((hour - 7) * 60) + mins,
			arrivals: arrivals,
		})
	}
	return samples, nil
}

type trainType string

const (
	trainTypeL4 trainType = "L4"
	trainTypeL8 trainType = "L8"
)

type scheduleEntry struct {
	TrainNum     int
	TrainType    trainType
	AArrivalTime int
	AAvailCap    int
	ABoarding    int
	BArrivalTime int
	BAvailCap    int
	BBoarding    int
	CArrivalTime int
	CAvailCap    int
	CBoarding    int
	UArrivalTime int
	UAvailCap    int
	UOffloading  int
}

const delay = 3
const AB = 8  // + delay?
const BC = 9  // + delay?
const CU = 11 // + delay?

func writeSchedule(entries []scheduleEntry, fpath string) error {
	outputf, err := os.Create(fpath)
	if err != nil {
		return err
	}
	defer outputf.Close()
	outEntries := make([][]string, 0, len(entries)+1)
	outEntries = append(outEntries, []string{
		"TrainNum",
		"TrainType",
		"A_ArrivalTime",
		"A_AvailCap",
		"A_Boarding",
		"B_ArrivalTime",
		"B_AvailCap",
		"B_Boarding",
		"C_ArrivalTime",
		"C_AvailCap",
		"C_Boarding",
		"U_ArrivalTime",
		"U_AvailCap",
		"U_Offloading",
	})
	for _, e := range entries {
		outEntries = append(outEntries, []string{
			fmt.Sprintf("%d", e.TrainNum),
			string(e.TrainType),
			fmt.Sprintf("%d:%02d", 7+(e.AArrivalTime/60), e.AArrivalTime%60),
			fmt.Sprintf("%d", e.AAvailCap),
			fmt.Sprintf("%d", e.ABoarding),
			fmt.Sprintf("%d:%02d", 7+(e.BArrivalTime/60), e.BArrivalTime%60),
			fmt.Sprintf("%d", e.BAvailCap),
			fmt.Sprintf("%d", e.BBoarding),
			fmt.Sprintf("%d:%02d", 7+(e.CArrivalTime/60), e.CArrivalTime%60),
			fmt.Sprintf("%d", e.CAvailCap),
			fmt.Sprintf("%d", e.CBoarding),
			fmt.Sprintf("%d:%02d", 7+(e.UArrivalTime/60), e.UArrivalTime%60),
			fmt.Sprintf("%d", e.UAvailCap),
			fmt.Sprintf("%d", e.UOffloading),
		})
	}
	w := csv.NewWriter(outputf)
	for _, e := range outEntries {
		if err := w.Write(e); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

type departure struct {
	ttype     trainType
	timestamp int
}

type pplDiff struct {
	timestamp int
	diff      int
}

type departureSchedule struct {
	arrivals   []arrivalSample
	departures []departure
}

// Make sure that departure schedule implements genome.
var _ eaopt.Genome = &departureSchedule{}

var maxScore = math.MaxFloat64

func (ds *departureSchedule) Evaluate() (float64, error) {
	if ds.departures[0].timestamp < 0 || ds.departures[15].timestamp > 60*3 {
		return maxScore, nil
	}
	for i := 1; i < 16; i++ {
		if ds.departures[i].timestamp-ds.departures[i-1].timestamp < 3 {
			return maxScore, nil
		}
	}
	schedule := generateSchedule(ds.arrivals, ds.departures)
	var (
		arrivals  int
		offloaded int
	)
	for _, a := range ds.arrivals {
		arrivals += a.arrivals
	}
	for _, e := range schedule {
		offloaded += e.UOffloading
	}
	if offloaded < arrivals {
		return maxScore, nil
	}
	score := scoreSchedule(ds.arrivals, schedule)
	return float64(score), nil
}

func (ds *departureSchedule) Mutate(rng *rand.Rand) {
	// Half the time, we swap two of the train sizes.
	// The other half of the time, we slightly increase and/or decrease
	// the departure times of our trains (~80% probability).
	opt := rng.Intn(2)
	if opt == 0 {
		var idx1 int
		var idx2 int
		for idx1 == idx2 {
			idx1, idx2 = rand.Intn(len(ds.departures)), rand.Intn(len(ds.departures))
		}
		tmpTtype := ds.departures[idx1].ttype
		ds.departures[idx1].ttype = ds.departures[idx2].ttype
		ds.departures[idx2].ttype = tmpTtype
	} else if opt == 1 {
		for i := 0; i < 16; i++ {
			ds.departures[i].timestamp += rand.Intn(4) - 2
		}
	}
	// Sanity check to make sure we're still in sorted order.
	// (As this is an assumption we make throughout the rest of the codebase)
	sort.Slice(ds.departures, func(i, j int) bool {
		return ds.departures[i].timestamp < ds.departures[j].timestamp
	})
}

func (ds *departureSchedule) Crossover(Y eaopt.Genome, rng *rand.Rand) {
	other := Y.(*departureSchedule)
	for i := 0; i < 16; i++ {
		ds.departures[i].timestamp = (ds.departures[i].timestamp + other.departures[i].timestamp) / 2
	}
	sort.Slice(ds.departures, func(i, j int) bool {
		return ds.departures[i].timestamp < ds.departures[j].timestamp
	})
}

func (ds *departureSchedule) Clone() eaopt.Genome {
	created := &departureSchedule{
		arrivals:   make([]arrivalSample, len(ds.arrivals)),
		departures: make([]departure, 16),
	}
	copy(created.arrivals, ds.arrivals)
	copy(created.departures, ds.departures)
	return created
}

func genomeMachine(arrivals []arrivalSample) func(*rand.Rand) eaopt.Genome {
	return func(_ *rand.Rand) eaopt.Genome {
		created := &departureSchedule{
			arrivals: arrivals,
			departures: []departure{
				{ttype: trainTypeL4, timestamp: 0},
				{ttype: trainTypeL8, timestamp: 10},
				{ttype: trainTypeL8, timestamp: 15},
				{ttype: trainTypeL8, timestamp: 20},
				{ttype: trainTypeL8, timestamp: 30},
				{ttype: trainTypeL8, timestamp: 40},
				{ttype: trainTypeL8, timestamp: 50},
				{ttype: trainTypeL8, timestamp: 60},
				{ttype: trainTypeL8, timestamp: 70},
				{ttype: trainTypeL8, timestamp: 80},
				{ttype: trainTypeL8, timestamp: 90},
				{ttype: trainTypeL8, timestamp: 105},
				{ttype: trainTypeL8, timestamp: 130},
				{ttype: trainTypeL4, timestamp: 150},
				{ttype: trainTypeL4, timestamp: 160},
				{ttype: trainTypeL4, timestamp: 180},
			},
		}
		return created
	}
}

var stationIndex = map[string]int{
	"A": 0,
	"B": 1,
	"C": 2,
}

var trainCapacity = map[trainType]int{
	trainTypeL4: 50 * 4,
	trainTypeL8: 50 * 8,
}

func generateSchedule(arrivals []arrivalSample, departures []departure) []scheduleEntry {
	// Keep track of arrivals and/or departures from station.
	pplArrivals := [3][]pplDiff{}
	for _, a := range arrivals {
		idx := stationIndex[a.station]
		pplArrivals[idx] = append(pplArrivals[idx], pplDiff{
			timestamp: a.minutes,
			diff:      a.arrivals,
		})
	}
	pplDepartures := [3][]pplDiff{}
	// A function which computes the number of people at a given station for time t.
	numPeople := func(station string, t int) int {
		var ppl int
		for _, a := range pplArrivals[stationIndex[station]] {
			if a.timestamp > t {
				break
			}
			ppl += a.diff
		}
		for _, d := range pplDepartures[stationIndex[station]] {
			if d.timestamp > t {
				break
			}
			ppl += d.diff
		}
		return ppl
	}
	// Calculate the schedule entries.
	entries := make([]scheduleEntry, 0, len(departures))
	for i, d := range departures {
		entry := scheduleEntry{
			TrainNum:     i + 1,
			TrainType:    d.ttype,
			AArrivalTime: d.timestamp,
			AAvailCap:    trainCapacity[d.ttype],
		}
		aBoarded := min(entry.AAvailCap, numPeople("A", d.timestamp+delay))
		entry.ABoarding = aBoarded
		pplDepartures[stationIndex["A"]] = append(pplDepartures[stationIndex["A"]], pplDiff{
			timestamp: d.timestamp + delay,
			diff:      -aBoarded,
		})
		entry.BArrivalTime = entry.AArrivalTime + delay + AB
		entry.BAvailCap = entry.AAvailCap - entry.ABoarding
		bBoarded := min(entry.BAvailCap, numPeople("B", entry.BArrivalTime))
		entry.BBoarding = bBoarded
		pplDepartures[stationIndex["B"]] = append(pplDepartures[stationIndex["B"]], pplDiff{
			timestamp: entry.BArrivalTime,
			diff:      -bBoarded,
		})
		entry.CArrivalTime = entry.BArrivalTime + delay + BC
		entry.CAvailCap = entry.BAvailCap - entry.BBoarding
		cBoarded := min(entry.CAvailCap, numPeople("C", entry.CArrivalTime))
		entry.CBoarding = cBoarded
		pplDepartures[stationIndex["C"]] = append(pplDepartures[stationIndex["C"]], pplDiff{
			timestamp: entry.CArrivalTime,
			diff:      -cBoarded,
		})
		entry.UArrivalTime = entry.CArrivalTime + delay + CU
		entry.UAvailCap = entry.CAvailCap - entry.CBoarding
		entry.UOffloading = aBoarded + bBoarded + cBoarded
		entries = append(entries, entry)
	}
	return entries
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func scoreSchedule(arrivals []arrivalSample, entries []scheduleEntry) float32 {
	parrivals := [3]map[int][]int{}
	for _, a := range arrivals {
		idx := stationIndex[a.station]
		for i := 0; i < a.arrivals; i++ {
			if parrivals[idx] == nil {
				parrivals[idx] = make(map[int][]int)
			}
			parrivals[idx][a.minutes] = append(parrivals[idx][a.minutes], a.minutes)
		}
	}
	tarrivals := [3]map[int]int{}
	for i := 0; i < 3; i++ {
		tarrivals[i] = make(map[int]int)
	}
	for _, e := range entries {
		tarrivals[0][e.AArrivalTime+delay] += e.AAvailCap
		tarrivals[1][e.BArrivalTime+delay] += e.BAvailCap
		tarrivals[2][e.CArrivalTime+delay] += e.CAvailCap
	}
	var (
		pbuf  = [3][]int{}
		take  []int
		m     int
		sum   int
		count int
	)
	for m < (60*3) || len(pbuf[0]) != 0 || len(pbuf[1]) != 0 || len(pbuf[2]) != 0 {
		for i := 0; i < 3; i++ {
			pbuf[i] = append(pbuf[i], parrivals[i][m]...)
			if v := tarrivals[i][m]; v != 0 {
				l := min(v, len(pbuf[i]))
				take, pbuf[i] = pbuf[i][:l], pbuf[i][l:]
				for _, t := range take {
					sum += m - t
				}
				count += len(take)
			}
		}
		m++ // todo: skip to next notable event?
	}
	return float32(sum) / float32(count)
}

func getOptimizedSchedule(arrivals []arrivalSample) ([]scheduleEntry, error) {
	ga, err := eaopt.NewDefaultGAConfig().NewGA()
	if err != nil {
		return nil, err
	}
	ga.NGenerations = 10
	if err := ga.Minimize(genomeMachine(arrivals)); err != nil {
		return nil, err
	}
	hof := ga.HallOfFame[0].Genome.(*departureSchedule)
	return generateSchedule(arrivals, hof.departures), nil
}

//go:embed index.html
var assets embed.FS

func schedulerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.ParseMultipartForm(10 << 20) // Max of 10MB
		file, _, err := r.FormFile("input")
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		defer file.Close()
		arrivals, err := getArrivals(file)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		schedule, err := getOptimizedSchedule(arrivals)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(schedule); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
}

func run() error {
	http.Handle("/", http.FileServer(http.FS(assets)))
	http.HandleFunc("/schedule", schedulerHandler())
	port := "8000"
	if p, ok := os.LookupEnv("PORT"); ok {
		port = p
	}
	return http.ListenAndServe(fmt.Sprintf(":%s", port), nil)
}
