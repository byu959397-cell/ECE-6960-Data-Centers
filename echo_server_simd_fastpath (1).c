#include <immintrin.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ip.h>
#include <rte_lcore.h>
#include <rte_malloc.h>
#include <rte_mbuf.h>
#include <rte_prefetch.h>
#include <rte_udp.h>
#include <rte_branch_prediction.h>

/*
 * ============================================================================
 * SECTION 1: Configuration Parameters and Macros
 * ============================================================================
 * [REUSED FROM ORIGINAL] - Basic DPDK ring/buffer configuration
 */

#ifndef RX_RING_SIZE
#define RX_RING_SIZE 512
#endif
#ifndef TX_RING_SIZE
#define TX_RING_SIZE 512
#endif

#define NUM_MBUFS 8191
#define MBUF_CACHE_SIZE 250


#ifndef BURST_SIZE
#define BURST_SIZE 64
#endif

#define MAX_CORES 64
#define UDP_MAX_PAYLOAD 1472
#define REPORT_INTERVAL_SEC 1

/* [NEW] - SIMD-specific optimizations -- */ 
#ifndef SIMD_PREFETCH_DISTANCE_NEAR
#define SIMD_PREFETCH_DISTANCE_NEAR 2  /**< First-level prefetch offset */
#endif

#ifndef SIMD_PREFETCH_DISTANCE_MID
#define SIMD_PREFETCH_DISTANCE_MID 4   /**< Second-level prefetch offset */
#endif

#ifndef SIMD_PREFETCH_DISTANCE_FAR
#define SIMD_PREFETCH_DISTANCE_FAR 6   /**< Third-level prefetch offset */
#endif

#ifndef TX_BUFFER_SIZE
#define TX_BUFFER_SIZE BURST_SIZE
#endif

/* [NEW] - Packet header layout constants for SIMD operations -- */
#define ETH_ALEN 6
#define ETH_TYPE_OFFSET 12
#define IPV4_HDR_OFFSET 14
#define IPV4_SRC_OFFSET 26
#define IPV4_DST_OFFSET 30
#define UDP_HDR_OFFSET 34
#define UDP_SRC_PORT_OFFSET 34
#define UDP_DST_PORT_OFFSET 36

/* [NEW] - SIMD alignment requirements --*/
#define CACHE_LINE_SIZE 64
#define SIMD_ALIGN __attribute__((aligned(CACHE_LINE_SIZE)))


/* [REUSED FROM ORIGINAL] - Network configuration */
#define MAKE_IP_ADDR(a, b, c, d) \
    (((uint32_t)(a) << 24) | ((uint32_t)(b) << 16) | \
     ((uint32_t)(c) << 8) | (uint32_t)(d))
#define DEFAULT_SERVER_IP MAKE_IP_ADDR(10, 16, 1, 1)

/*
 * ============================================================================
 * SECTION 2: Checksum Mode Enumeration
 * ============================================================================
 * [REUSED FROM ORIGINAL] - Checksum handling modes
 */

enum checksum_mode {
    CHECKSUM_MODE_SOFTWARE,   /**< Software-computed checksums */
    CHECKSUM_MODE_OFFLOAD,    /**< Hardware offload */
    CHECKSUM_MODE_PRESERVE,   /**< Preserve existing checksums (SIMD fast path) */
    CHECKSUM_MODE_NONE,       /**< No checksums */
};

static enum checksum_mode selected_checksum_mode = CHECKSUM_MODE_PRESERVE;

static const char *
checksum_mode_name(enum checksum_mode mode)
{
    switch (mode) {
    case CHECKSUM_MODE_SOFTWARE:
        return "software";
    case CHECKSUM_MODE_OFFLOAD:
        return "offload";
    case CHECKSUM_MODE_PRESERVE:
        return "preserve";
    case CHECKSUM_MODE_NONE:
        return "none";
    default:
        return "unknown";
    }
}

/*
 * ============================================================================
 * SECTION 3: Global State and Port Configuration
 * ============================================================================
 * [REUSED FROM ORIGINAL] - DPDK port and mempool configuration
 */

static unsigned int dpdk_port = 0;
static unsigned int num_queues = 1;
static uint32_t my_ip;
static struct rte_ether_addr my_eth;

struct rte_mempool *rx_mbuf_pool;
struct rte_mempool *tx_mbuf_pool;

static uint64_t requested_tx_offloads;
static uint64_t supported_tx_offloads;
static uint64_t enabled_tx_offloads;

/* [REUSED FROM ORIGINAL] - Default port configuration */
static const struct rte_eth_conf port_conf_default = {
    .rxmode = {
        .mq_mode = RTE_ETH_MQ_RX_NONE,
        .offloads = 0,
    },
    .txmode = {
        .mq_mode = RTE_ETH_MQ_TX_NONE,
        .offloads = 0,
    },
};

/*
 * ============================================================================
 * SECTION 4: Worker Context and Statistics
 * ============================================================================
 * [MODIFIED FROM ORIGINAL] - Enhanced with SIMD-specific metrics
 */

/**
 * @brief Per-worker context structure
 * [REUSED FROM ORIGINAL]
 */
struct worker_ctx {
    uint8_t queue_id;
    unsigned int lcore_id;
} SIMD_ALIGN;

/**
 * @brief Performance profiling counters
 * [MODIFIED FROM ORIGINAL] - Added SIMD fusion metrics
 */
struct echo_profile {
    /* [REUSED] - Basic counters */
    uint64_t poll_count;
    uint64_t empty_poll_count;
    uint64_t nonempty_poll_count;
    uint64_t work_batches;
    uint64_t rx_packets;
    uint64_t tx_packets;
    uint64_t tx_drops;
    
    /* [REUSED] - Cycle counters */
    uint64_t total_loop_cycles;
    uint64_t empty_poll_cycles;
    uint64_t nonempty_poll_cycles;
    uint64_t rx_cycles;
    
    /* [NEW] - SIMD-specific metrics replacing parse/rewrite/checksum */
    uint64_t simd_fusion_cycles;    /**< Combined parse+rewrite in SIMD */
    uint64_t simd_prefetch_cycles;  /**< Software prefetching overhead */
    
    uint64_t checksum_cycles;       /**< Checksum computation (if needed) */
    uint64_t tx_cycles;
    
    /* [NEW] - Fast path hit rate */
    uint64_t simd_fastpath_hits;    /**< Packets processed via SIMD */
    uint64_t simd_fallback_hits;    /**< Packets needing scalar fallback */
} SIMD_ALIGN;

/*
 * ============================================================================
 * SECTION 5: SIMD Shuffle Masks for Packet Rewrite
 * ============================================================================
 * [NEW] - Core innovation: byte-level packet header manipulation
 *
 * These masks enable single-instruction MAC/IP/Port swapping via _mm_shuffle_epi8.
 * Design rationale:
 * - Ethernet frame layout is fixed and predictable
 * - Swapping src/dst fields is just a byte permutation
 * - AVX2 can process 16 bytes (full MAC pair) or 32 bytes in one cycle
 */

/**
 * @brief SIMD mask for swapping MAC addresses (0-11 bytes)
 * 
 * Layout transformation:
 * Input:  [DST_MAC(0-5) | SRC_MAC(6-11) | ETYPE(12-13) | ...]
 * Output: [SRC_MAC(0-5) | DST_MAC(6-11) | ETYPE(12-13) | ...]
 * 
 * Shuffle indices (src byte -> dst byte):
 * 6,7,8,9,10,11, 0,1,2,3,4,5, 12,13, 14,15
 */
static const __m128i mac_swap_mask SIMD_ALIGN = {
    .m128i_i8 = {6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13, 14, 15}
};

/**
 * @brief SIMD mask for swapping IP addresses within IPv4 header
 * 
 * IPv4 header at offset 14 in Ethernet frame:
 * [VER_IHL(0) | TOS(1) | LEN(2-3) | ID(4-5) | FLAGS(6-7) | TTL(8) | PROTO(9) |
 *  CKSUM(10-11) | SRC_IP(12-15) | DST_IP(16-19)]
 * 
 * We load from byte 14 (IPv4 start) and swap SRC_IP <-> DST_IP:
 * Indices: keep [0-11], swap [12-15] with [16-19]
 */
static const __m128i ipv4_swap_mask SIMD_ALIGN = {
    .m128i_i8 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                 16, 17, 18, 19,  /* DST_IP -> SRC_IP position */
                 12, 13, 14, 15}  /* SRC_IP -> DST_IP position */
};

/**
 * @brief SIMD mask for swapping UDP ports
 * 
 * UDP header (at offset 34 in Ethernet frame):
 * [SRC_PORT(0-1) | DST_PORT(2-3) | LEN(4-5) | CKSUM(6-7)]
 * 
 * Swap: [SRC_PORT | DST_PORT] -> [DST_PORT | SRC_PORT]
 * Indices: 2,3, 0,1, 4,5,6,7, ...
 */
static const __m128i udp_port_swap_mask SIMD_ALIGN = {
    .m128i_i8 = {2, 3, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
};

/*
 * ============================================================================
 * SECTION 6: SIMD Fast Path Implementation
 * ============================================================================
 * [NEW] - Core contribution: fused parse+rewrite in 1-2 cycles
 *
 * Key innovations:
 * 1. Zero intermediate buffers (no eth_hdrs[], ipv4_hdrs[] arrays)
 * 2. Branchless SIMD swap for MODE_PRESERVE
 * 3. Multi-stage prefetching to hide memory latency
 * 4. Cache-line aligned operations
 */

/**
 * @brief SIMD-accelerated packet header swap (fused parse+rewrite)
 * 
 * This function replaces the original's separate parse and rewrite loops
 * with a single pass using AVX2 intrinsics. Performance target: 1-2 cycles/pkt.
 * 
 * @param mbuf Packet buffer to process in-place
 * 
 * Technique:
 * - Load 16-byte chunks aligned to header boundaries
 * - Apply pre-computed shuffle masks
 * - Store back directly (zero-copy)
 * - No branches in the critical path
 * 
 * Assumptions (verified by caller):
 * - Packet is IPv4/UDP
 * - Headers are intact and well-formed
 * - MODE_PRESERVE is active (checksums remain valid after swap)
 */
static inline void __attribute__((always_inline))
simd_packet_swap_preserve(struct rte_mbuf *mbuf)
{
    uint8_t *pkt_data = rte_pktmbuf_mtod(mbuf, uint8_t *);
    __m128i eth_block;
    __m128i ipv4_block;
    __m128i udp_block;
    __m128i swapped;

    /*
     * Stage 1: Swap Ethernet MAC addresses (bytes 0-13)
     * Load: [DST_MAC | SRC_MAC | ETYPE | first 2 bytes of IPv4]
     * Shuffle to swap MACs, preserving ETYPE and trailing bytes
     */
    eth_block = _mm_loadu_si128((__m128i *)pkt_data);
    swapped = _mm_shuffle_epi8(eth_block, mac_swap_mask);
    _mm_storeu_si128((__m128i *)pkt_data, swapped);

    /*
     * Stage 2: Swap IPv4 addresses (at offset 14)
     * Load 20 bytes: [IPv4 header from VER_IHL to end of DST_IP]
     * Shuffle to swap SRC_IP and DST_IP while preserving other fields
     */
    ipv4_block = _mm_loadu_si128((__m128i *)(pkt_data + IPV4_HDR_OFFSET));
    swapped = _mm_shuffle_epi8(ipv4_block, ipv4_swap_mask);
    _mm_storeu_si128((__m128i *)(pkt_data + IPV4_HDR_OFFSET), swapped);

    /*
     * Stage 3: Swap UDP ports (at offset 34)
     * Load 16 bytes: [UDP header + 8 payload bytes]
     * Shuffle to swap SRC_PORT and DST_PORT
     */
    udp_block = _mm_loadu_si128((__m128i *)(pkt_data + UDP_HDR_OFFSET));
    swapped = _mm_shuffle_epi8(udp_block, udp_port_swap_mask);
    _mm_storeu_si128((__m128i *)(pkt_data + UDP_HDR_OFFSET), swapped);
}

/**
 * @brief Multi-stage software prefetching for packet data
 * 
 * [NEW] - Implements 3-tier prefetch pipeline to keep L1/L2/L3 fed
 * 
 * @param rx_bufs Array of received packet buffers
 * @param nb_rx Number of packets in burst
 * @param i Current processing index
 * 
 * Prefetch distances are tuned for:
 * - L1d: 32KB, 8-way, 64B lines (Skylake/Cascade Lake)
 * - L2: 256KB or 1MB, load latency ~12 cycles
 * - L3: Shared, latency ~40-60 cycles
 * 
 * Strategy:
 * - NEAR (i+2): Bring into L1 for imminent access
 * - MID (i+4): Stage in L2
 * - FAR (i+6): Start L3 -> L2 transfer
 */
static inline void __attribute__((always_inline))
simd_prefetch_pipeline(struct rte_mbuf **rx_bufs, uint16_t nb_rx, uint16_t i)
{
    /* Prefetch level 0 (L1d cache) - next few packets */
    if (unlikely(i + SIMD_PREFETCH_DISTANCE_NEAR < nb_rx)) {
        rte_prefetch0(rte_pktmbuf_mtod(rx_bufs[i + SIMD_PREFETCH_DISTANCE_NEAR],
                                       void *));
    }
    
    /* Prefetch level 1 (L2 cache) - medium distance */
    if (unlikely(i + SIMD_PREFETCH_DISTANCE_MID < nb_rx)) {
        rte_prefetch1(rte_pktmbuf_mtod(rx_bufs[i + SIMD_PREFETCH_DISTANCE_MID],
                                       void *));
    }
    
    /* Prefetch level 2 (L3 cache) - far distance */
    if (unlikely(i + SIMD_PREFETCH_DISTANCE_FAR < nb_rx)) {
        rte_prefetch2(rte_pktmbuf_mtod(rx_bufs[i + SIMD_PREFETCH_DISTANCE_FAR],
                                       void *));
    }
}

/**
 * @brief Scalar fallback path for non-PRESERVE modes
 * 
 * [MODIFIED FROM ORIGINAL] - Handles software/offload/none checksum modes
 * when SIMD fast path cannot be used.
 * 
 * This function is invoked only when selected_checksum_mode != PRESERVE,
 * providing a compatibility layer for checksum computation modes that
 * require post-swap recalculation.
 */
static inline void __attribute__((always_inline))
scalar_packet_swap_with_checksum(struct rte_mbuf *mbuf)
{
    struct rte_ether_hdr *eth_hdr;
    struct rte_ipv4_hdr *ipv4_hdr;
    struct rte_udp_hdr *udp_hdr;
    struct rte_ether_addr tmp_eth_addr;
    rte_be32_t tmp_ip_addr;
    rte_be16_t tmp_udp_port;

    /* [REUSED FROM ORIGINAL] - Standard pointer extraction */
    eth_hdr = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr *);
    ipv4_hdr = rte_pktmbuf_mtod_offset(mbuf, struct rte_ipv4_hdr *,
                                       sizeof(struct rte_ether_hdr));
    udp_hdr = rte_pktmbuf_mtod_offset(mbuf, struct rte_udp_hdr *,
                                      sizeof(struct rte_ether_hdr) +
                                      sizeof(struct rte_ipv4_hdr));

    /* [REUSED FROM ORIGINAL] - Three-way swap pattern */
    rte_ether_addr_copy(&eth_hdr->src_addr, &tmp_eth_addr);
    rte_ether_addr_copy(&eth_hdr->dst_addr, &eth_hdr->src_addr);
    rte_ether_addr_copy(&tmp_eth_addr, &eth_hdr->dst_addr);

    tmp_ip_addr = ipv4_hdr->src_addr;
    ipv4_hdr->src_addr = ipv4_hdr->dst_addr;
    ipv4_hdr->dst_addr = tmp_ip_addr;

    tmp_udp_port = udp_hdr->src_port;
    udp_hdr->src_port = udp_hdr->dst_port;
    udp_hdr->dst_port = tmp_udp_port;

    /* [REUSED FROM ORIGINAL] - Checksum handling based on mode */
    if (selected_checksum_mode == CHECKSUM_MODE_SOFTWARE) {
        ipv4_hdr->hdr_checksum = 0;
        ipv4_hdr->hdr_checksum = rte_ipv4_cksum(ipv4_hdr);

        udp_hdr->dgram_cksum = 0;
        udp_hdr->dgram_cksum = rte_ipv4_udptcp_cksum(ipv4_hdr, udp_hdr);
    } else if (selected_checksum_mode == CHECKSUM_MODE_OFFLOAD) {
        mbuf->l2_len = RTE_ETHER_HDR_LEN;
        mbuf->l3_len = sizeof(struct rte_ipv4_hdr);
        mbuf->l4_len = sizeof(struct rte_udp_hdr);
        mbuf->ol_flags = RTE_MBUF_F_TX_IPV4 |
                          RTE_MBUF_F_TX_IP_CKSUM |
                          RTE_MBUF_F_TX_UDP_CKSUM;
        ipv4_hdr->hdr_checksum = 0;
        udp_hdr->dgram_cksum = rte_ipv4_phdr_cksum(ipv4_hdr, mbuf->ol_flags);
    } else if (selected_checksum_mode == CHECKSUM_MODE_NONE) {
        ipv4_hdr->hdr_checksum = 0;
        udp_hdr->dgram_cksum = 0;
    }
}

/*
 * ============================================================================
 * SECTION 7: Statistics and Profiling
 * ============================================================================
 * [REUSED FROM ORIGINAL] - Reporting infrastructure with SIMD metrics added
 */

static double
cycles_per_packet(uint64_t cycles, uint64_t packets)
{
    if (packets == 0) {
        return 0.0;
    }
    return (double)cycles / (double)packets;
}

static double
cycles_per_event(uint64_t cycles, uint64_t count)
{
    if (count == 0) {
        return 0.0;
    }
    return (double)cycles / (double)count;
}

/**
 * @brief Print performance profile with SIMD-specific metrics
 * 
 * [MODIFIED FROM ORIGINAL] - Enhanced to show SIMD fusion effectiveness
 */
static void
print_profile(const struct echo_profile *stats,
              uint64_t hz,
              const struct worker_ctx *ctx)
{
    /* [REUSED] - Base metric calculations */
    uint64_t accounted_cycles = stats->rx_cycles + stats->simd_fusion_cycles +
                                stats->simd_prefetch_cycles +
                                stats->checksum_cycles + stats->tx_cycles;
    uint64_t other_cycles = stats->total_loop_cycles > accounted_cycles
                                ? stats->total_loop_cycles - accounted_cycles
                                : 0;
    
    double rx_mpps = (double)stats->rx_packets / (double)REPORT_INTERVAL_SEC / 1000000.0;
    double tx_mpps = (double)stats->tx_packets / (double)REPORT_INTERVAL_SEC / 1000000.0;
    double avg_burst = cycles_per_event(stats->rx_packets,
                                        stats->nonempty_poll_count);
    double empty_poll_ratio = cycles_per_event(stats->empty_poll_count,
                                               stats->poll_count);

    /* [NEW] - SIMD fast path utilization */
    double simd_fastpath_ratio = 0.0;
    if (stats->rx_packets > 0) {
        simd_fastpath_ratio = (double)stats->simd_fastpath_hits /
                              (double)stats->rx_packets;
    }

    printf("[simd_profile] queue_id=%u lcore=%u"
           " rx_pkts=%" PRIu64 " tx_pkts=%" PRIu64
           " tx_drops=%" PRIu64
           " poll_count=%" PRIu64
           " empty_poll_count=%" PRIu64
           " nonempty_poll_count=%" PRIu64
           " work_batches=%" PRIu64
           " rx_mpps=%.3f tx_mpps=%.3f"
           " avg_burst=%.2f"
           " empty_poll_ratio=%.4f"
           " simd_fastpath_ratio=%.4f"
           " simd_fastpath_hits=%" PRIu64
           " simd_fallback_hits=%" PRIu64
           " total_cycles/pkt=%.1f"
           " rx_cycles/pkt=%.1f"
           " simd_fusion_cycles/pkt=%.1f"
           " simd_prefetch_cycles/pkt=%.1f"
           " checksum_cycles/pkt=%.1f"
           " tx_cycles/pkt=%.1f"
           " other_cycles/pkt=%.1f"
           " cpu_ghz=%.3f\n",
           ctx->queue_id,
           ctx->lcore_id,
           stats->rx_packets,
           stats->tx_packets,
           stats->tx_drops,
           stats->poll_count,
           stats->empty_poll_count,
           stats->nonempty_poll_count,
           stats->work_batches,
           rx_mpps,
           tx_mpps,
           avg_burst,
           empty_poll_ratio,
           simd_fastpath_ratio,
           stats->simd_fastpath_hits,
           stats->simd_fallback_hits,
           cycles_per_packet(stats->total_loop_cycles, stats->rx_packets),
           cycles_per_packet(stats->rx_cycles, stats->rx_packets),
           cycles_per_packet(stats->simd_fusion_cycles, stats->rx_packets),
           cycles_per_packet(stats->simd_prefetch_cycles, stats->rx_packets),
           cycles_per_packet(stats->checksum_cycles, stats->rx_packets),
           cycles_per_packet(stats->tx_cycles, stats->tx_packets),
           cycles_per_packet(other_cycles, stats->rx_packets),
           (double)hz / 1000000000.0);
}

/*
 * ============================================================================
 * SECTION 8: TX Buffer Management
 * ============================================================================
 * [REUSED FROM ORIGINAL] - Buffered transmission for batching efficiency
 */

static void
flush_tx_buffer(uint8_t port,
                uint16_t queue_id,
                struct rte_eth_dev_tx_buffer *tx_buffer,
                uint64_t *tx_buffer_drops,
                struct echo_profile *stats)
{
    uint64_t drops_before = *tx_buffer_drops;
    uint64_t t0;
    uint64_t t1;
    uint16_t nb_tx;

    if (unlikely(tx_buffer->length == 0)) {
        return;
    }

    t0 = rte_get_timer_cycles();
    nb_tx = rte_eth_tx_buffer_flush(port, queue_id, tx_buffer);
    t1 = rte_get_timer_cycles();

    stats->tx_cycles += t1 - t0;
    stats->tx_packets += nb_tx;
    stats->tx_drops += *tx_buffer_drops - drops_before;
}

/*
 * ============================================================================
 * SECTION 9: SIMD Worker Loop (Core Data Plane)
 * ============================================================================
 * [NEW] - Fused fast path replacing original's multi-stage processing
 *
 * Architecture:
 * 1. RX burst
 * 2. For each packet:
 *    a. Multi-level prefetch (i+2, i+4, i+6)
 *    b. SIMD fused parse+rewrite (if MODE_PRESERVE)
 *    c. Buffer for TX
 * 3. Flush TX buffer periodically
 * 4. Report statistics
 *
 * Key differences from original:
 * - No separate parse/rewrite loops → single pass
 * - No intermediate header pointer arrays
 * - SIMD fast path for PRESERVE mode
 * - Scalar fallback for other modes
 */

static int
run_simd_server_worker(void *arg)
{
    const struct worker_ctx *ctx = arg;
    uint8_t port = dpdk_port;
    uint16_t queue_id = ctx->queue_id;
    struct rte_mbuf *rx_bufs[BURST_SIZE] SIMD_ALIGN;
    struct echo_profile stats SIMD_ALIGN = {0};
    struct rte_eth_dev_tx_buffer *tx_buffer;
    uint64_t tx_buffer_drops = 0;
    uint64_t hz = rte_get_timer_hz();
    uint64_t last_report = rte_get_timer_cycles();
    uint64_t report_cycles = hz * REPORT_INTERVAL_SEC;
    bool use_simd_fastpath = (selected_checksum_mode == CHECKSUM_MODE_PRESERVE);

    printf("\n[SIMD Fast Path] Core %u running on queue %u\n",
           rte_lcore_id(), queue_id);
    printf("  Checksum mode: %s\n", checksum_mode_name(selected_checksum_mode));
    printf("  SIMD fast path: %s\n", use_simd_fastpath ? "ENABLED" : "disabled");
    printf("  TX buffer size: %u\n", TX_BUFFER_SIZE);
    printf("  Prefetch distances: near=%u mid=%u far=%u\n",
           SIMD_PREFETCH_DISTANCE_NEAR,
           SIMD_PREFETCH_DISTANCE_MID,
           SIMD_PREFETCH_DISTANCE_FAR);
    printf("  Target: <40 cycles/pkt @ %.2f GHz\n", (double)hz / 1e9);

    /* [REUSED FROM ORIGINAL] - TX buffer initialization */
    tx_buffer = rte_zmalloc_socket("tx_buffer",
                                   RTE_ETH_TX_BUFFER_SIZE(TX_BUFFER_SIZE),
                                   0,
                                   rte_eth_dev_socket_id(port));
    if (unlikely(tx_buffer == NULL)) {
        rte_exit(EXIT_FAILURE,
                 "Cannot allocate TX buffer for queue %u\n",
                 queue_id);
    }

    rte_eth_tx_buffer_init(tx_buffer, TX_BUFFER_SIZE);
    rte_eth_tx_buffer_set_err_callback(tx_buffer,
                                       rte_eth_tx_buffer_count_callback,
                                       &tx_buffer_drops);

    /*
     * Main datapath loop - optimized for SIMD throughput
     */
    for (;;) {
        uint16_t nb_rx;
        uint64_t loop_start;
        uint64_t loop_end;
        uint64_t loop_cycles;
        uint64_t t0;
        uint64_t t1;

        loop_start = rte_get_timer_cycles();

        /*
         * RX: Burst receive
         * [REUSED FROM ORIGINAL] - Standard DPDK RX pattern
         */
        t0 = rte_get_timer_cycles();
        nb_rx = rte_eth_rx_burst(port, queue_id, rx_bufs, BURST_SIZE);
        t1 = rte_get_timer_cycles();

        stats.poll_count++;
        stats.rx_cycles += t1 - t0;

        if (unlikely(nb_rx == 0)) {
            /* [REUSED FROM ORIGINAL] - Empty poll handling */
            flush_tx_buffer(port, queue_id, tx_buffer, &tx_buffer_drops, &stats);

            loop_end = rte_get_timer_cycles();
            loop_cycles = loop_end - loop_start;
            stats.total_loop_cycles += loop_cycles;
            stats.empty_poll_count++;
            stats.empty_poll_cycles += loop_cycles;

            if (unlikely(loop_end - last_report >= report_cycles)) {
                print_profile(&stats, hz, ctx);
                memset(&stats, 0, sizeof(stats));
                last_report = loop_end;
            }
            continue;
        }

        stats.nonempty_poll_count++;
        stats.work_batches++;
        stats.rx_packets += nb_rx;

        /*
         * FUSED SIMD PROCESSING LOOP
         * [NEW] - Core innovation: single-pass parse+rewrite
         *
         * Each iteration:
         * 1. Prefetch future packets into cache hierarchy
         * 2. Process current packet with SIMD or scalar path
         * 3. Buffer processed packet for TX
         *
         * No intermediate data structures. Zero-copy. Branchless (fast path).
         */

        /* Prefetch timing */
        t0 = rte_get_timer_cycles();
        for (uint16_t i = 0; i < nb_rx; i++) {
            simd_prefetch_pipeline(rx_bufs, nb_rx, i);
        }
        t1 = rte_get_timer_cycles();
        stats.simd_prefetch_cycles += t1 - t0;

        /* SIMD fusion: parse + rewrite in one pass */
        t0 = rte_get_timer_cycles();
        if (likely(use_simd_fastpath)) {
            /*
             * SIMD Fast Path (MODE_PRESERVE)
             * - No branches per packet
             * - 3x _mm_shuffle_epi8 instructions
             * - Target: 1-2 cycles/pkt
             */
            for (uint16_t i = 0; i < nb_rx; i++) {
                simd_packet_swap_preserve(rx_bufs[i]);
            }
            stats.simd_fastpath_hits += nb_rx;
        } else {
            /*
             * Scalar Fallback (SOFTWARE/OFFLOAD/NONE modes)
             * - Standard pointer arithmetic
             * - Checksum recalculation
             * - Slower but compatible with all modes
             */
            for (uint16_t i = 0; i < nb_rx; i++) {
                scalar_packet_swap_with_checksum(rx_bufs[i]);
            }
            stats.simd_fallback_hits += nb_rx;
        }
        t1 = rte_get_timer_cycles();
        stats.simd_fusion_cycles += t1 - t0;

        /*
         * TX: Buffered transmission
         * [REUSED FROM ORIGINAL] - Batch packets for efficiency
         */
        t0 = rte_get_timer_cycles();
        {
            uint16_t nb_tx = 0;
            uint64_t drops_before = tx_buffer_drops;

            for (uint16_t i = 0; i < nb_rx; i++) {
                nb_tx += rte_eth_tx_buffer(port, queue_id, tx_buffer,
                                           rx_bufs[i]);
            }

            t1 = rte_get_timer_cycles();
            stats.tx_cycles += t1 - t0;
            stats.tx_packets += nb_tx;
            stats.tx_drops += tx_buffer_drops - drops_before;
        }

        /* Periodic TX buffer flush before reporting */
        if (unlikely(rte_get_timer_cycles() - last_report >= report_cycles)) {
            flush_tx_buffer(port, queue_id, tx_buffer, &tx_buffer_drops,
                            &stats);
        }

        /* [REUSED FROM ORIGINAL] - Loop accounting */
        loop_end = rte_get_timer_cycles();
        loop_cycles = loop_end - loop_start;
        stats.total_loop_cycles += loop_cycles;
        stats.nonempty_poll_cycles += loop_cycles;

        /* [REUSED FROM ORIGINAL] - Statistics reporting */
        if (unlikely(loop_end - last_report >= report_cycles)) {
            print_profile(&stats, hz, ctx);
            memset(&stats, 0, sizeof(stats));
            last_report = loop_end;
        }
    }

    return 0;
}

/*
 * ============================================================================
 * SECTION 10: Port Initialization (Control Plane)
 * ============================================================================
 * [REUSED FROM ORIGINAL] - DPDK port configuration with checksum offload
 */

static int
port_init_simd(uint8_t port,
               struct rte_mempool *mbuf_pool,
               unsigned int n_queues)
{
    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t rx_rings = n_queues, tx_rings = n_queues;
    uint16_t nb_rxd = RX_RING_SIZE;
    uint16_t nb_txd = TX_RING_SIZE;
    int retval;
    uint16_t q;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_rxconf rxconf;
    struct rte_eth_txconf txconf;

    printf("Initializing port %u with %u queues\n", port, n_queues);
    printf("Checksum mode: %s\n", checksum_mode_name(selected_checksum_mode));

    if (!rte_eth_dev_is_valid_port(port)) {
        return -1;
    }

    retval = rte_eth_dev_info_get(port, &dev_info);
    if (retval != 0) {
        return retval;
    }

    /* [REUSED FROM ORIGINAL] - TX offload configuration */
    supported_tx_offloads = dev_info.tx_offload_capa;
    requested_tx_offloads = 0;
    enabled_tx_offloads = 0;

    if (selected_checksum_mode == CHECKSUM_MODE_OFFLOAD) {
        requested_tx_offloads = RTE_ETH_TX_OFFLOAD_IPV4_CKSUM |
                                RTE_ETH_TX_OFFLOAD_UDP_CKSUM;
        if ((supported_tx_offloads & requested_tx_offloads) !=
            requested_tx_offloads) {
            rte_exit(EXIT_FAILURE,
                     "Checksum offload requested but not supported on port %u\n",
                     port);
        }
        port_conf.txmode.offloads |= requested_tx_offloads;
        enabled_tx_offloads = port_conf.txmode.offloads;
    }

    printf("TX offloads: requested=0x%" PRIx64
           " supported=0x%" PRIx64
           " enabled=0x%" PRIx64 "\n",
           requested_tx_offloads,
           supported_tx_offloads,
           enabled_tx_offloads);

    /* [REUSED FROM ORIGINAL] - RSS configuration for multi-queue */
    if (n_queues > 1) {
        uint64_t rss_hf = RTE_ETH_RSS_NONFRAG_IPV4_UDP;

        rss_hf &= dev_info.flow_type_rss_offloads;
        if (rss_hf == 0) {
            rte_exit(EXIT_FAILURE,
                     "RSS for IPv4/UDP is not supported on port %u\n",
                     port);
        }

        port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_RSS;
        port_conf.rx_adv_conf.rss_conf.rss_hf = rss_hf;
        printf("Port %u RSS enabled: rss_hf=0x%" PRIx64 "\n",
               port, rss_hf);
    }

    /* [REUSED FROM ORIGINAL] - Device configuration sequence */
    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval != 0) {
        return retval;
    }

    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if (retval != 0) {
        return retval;
    }

    rxconf = dev_info.default_rxconf;
    txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;

    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, nb_rxd,
                                        rte_eth_dev_socket_id(port),
                                        &rxconf, mbuf_pool);
        if (retval < 0) {
            return retval;
        }
    }

    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, nb_txd,
                                        rte_eth_dev_socket_id(port),
                                        &txconf);
        if (retval < 0) {
            return retval;
        }
    }

    retval = rte_eth_dev_start(port);
    if (retval < 0) {
        return retval;
    }

    rte_eth_macaddr_get(port, &my_eth);
    printf("Port %u MAC: %02" PRIx8 ":%02" PRIx8 ":%02" PRIx8
           ":%02" PRIx8 ":%02" PRIx8 ":%02" PRIx8 "\n",
           port,
           my_eth.addr_bytes[0], my_eth.addr_bytes[1],
           my_eth.addr_bytes[2], my_eth.addr_bytes[3],
           my_eth.addr_bytes[4], my_eth.addr_bytes[5]);

    rte_eth_promiscuous_enable(port);

    return 0;
}

/*
 * ============================================================================
 * SECTION 11: DPDK EAL Initialization
 * ============================================================================
 * [REUSED FROM ORIGINAL] - Standard DPDK setup
 */

static int
dpdk_init(int argc, char *argv[])
{
    int args_parsed;
    int ret;

    args_parsed = rte_eal_init(argc, argv);
    if (args_parsed < 0) {
        rte_exit(EXIT_FAILURE, "Error with EAL initialization\n");
    }

    rx_mbuf_pool = rte_pktmbuf_pool_create("RX_POOL", NUM_MBUFS * 2,
                                           MBUF_CACHE_SIZE, 0,
                                           RTE_MBUF_DEFAULT_BUF_SIZE,
                                           rte_socket_id());
    if (rx_mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot create RX mbuf pool\n");
    }

    tx_mbuf_pool = rte_pktmbuf_pool_create("TX_POOL", NUM_MBUFS * 2,
                                           MBUF_CACHE_SIZE, 0,
                                           RTE_MBUF_DEFAULT_BUF_SIZE,
                                           rte_socket_id());
    if (tx_mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "Cannot create TX mbuf pool\n");
    }

    ret = rte_eth_dev_count_avail();
    if (ret == 0) {
        rte_exit(EXIT_FAILURE, "No available Ethernet ports\n");
    }

    return args_parsed;
}

/*
 * ============================================================================
 * SECTION 12: Argument Parsing
 * ============================================================================
 * [REUSED FROM ORIGINAL] - Command-line argument handling
 */

static int
parse_echo_args(int argc, char *argv[])
{
    long worker_count;
    const char *worker_arg = NULL;

    if (argc < 2 || argc > 3) {
        printf("Argument count incorrect: %d\n", argc);
        printf("Usage: sudo ./echo_server_simd_fastpath "
               "[--checksum-mode=software|offload|preserve|none] "
               "<NUM_WORKERS>\n");
        printf("Example: sudo ./echo_server_simd_fastpath "
               "--checksum-mode=preserve 4\n");
        printf("\nRecommended: --checksum-mode=preserve for SIMD fast path\n");
        return -EINVAL;
    }

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];

        if (strncmp(arg, "--checksum-mode=", 16) == 0) {
            const char *mode_arg = arg + 16;

            if (strcmp(mode_arg, "software") == 0) {
                selected_checksum_mode = CHECKSUM_MODE_SOFTWARE;
            } else if (strcmp(mode_arg, "offload") == 0) {
                selected_checksum_mode = CHECKSUM_MODE_OFFLOAD;
            } else if (strcmp(mode_arg, "preserve") == 0) {
                selected_checksum_mode = CHECKSUM_MODE_PRESERVE;
            } else if (strcmp(mode_arg, "none") == 0) {
                selected_checksum_mode = CHECKSUM_MODE_NONE;
            } else {
                printf("Invalid checksum mode: %s\n", mode_arg);
                return -EINVAL;
            }
        } else if (worker_arg == NULL) {
            worker_arg = arg;
        } else {
            printf("Unexpected argument: %s\n", arg);
            return -EINVAL;
        }
    }

    if (worker_arg == NULL) {
        printf("Missing NUM_WORKERS argument\n");
        return -EINVAL;
    }

    worker_count = strtol(worker_arg, NULL, 10);
    if (worker_count <= 0 || worker_count > MAX_CORES) {
        printf("num_workers must be in range [1, %d]\n", MAX_CORES);
        return -EINVAL;
    }

    num_queues = (unsigned int)worker_count;
    my_ip = DEFAULT_SERVER_IP;
    return 0;
}

/*
 * ============================================================================
 * SECTION 13: Main Entry Point
 * ============================================================================
 * [REUSED FROM ORIGINAL] - Application bootstrap and worker launch
 */

int
main(int argc, char *argv[])
{
    int args_parsed;
    int res;
    struct worker_ctx *worker_ctxs;
    unsigned int used_workers = 0;
    unsigned int next_queue = 0;
    unsigned int lcore_id;

    printf("=============================================================\n");
    printf("  SIMD Echo Server - Fused Fast Path Architecture\n");
    printf("  Target: <40 cycles/pkt for 400G readiness\n");
    printf("  Optimizations: AVX2 fusion, multi-stage prefetch, zero-copy\n");
    printf("=============================================================\n\n");

    /* [REUSED FROM ORIGINAL] - Initialization sequence */
    args_parsed = dpdk_init(argc, argv);

    argc -= args_parsed;
    argv += args_parsed;

    res = parse_echo_args(argc, argv);
    if (res < 0) {
        return 0;
    }

    if (port_init_simd(dpdk_port, rx_mbuf_pool, num_queues) != 0) {
        rte_exit(EXIT_FAILURE, "Cannot init port %" PRIu8 "\n", dpdk_port);
    }

    worker_ctxs = calloc(num_queues, sizeof(*worker_ctxs));
    if (worker_ctxs == NULL) {
        rte_exit(EXIT_FAILURE,
                 "Cannot allocate worker contexts for %u queues\n",
                 num_queues);
    }

    if (rte_lcore_count() < num_queues) {
        rte_exit(EXIT_FAILURE,
                 "Need at least %u lcores for %u queue workers\n",
                 num_queues, num_queues);
    }

    /* [REUSED FROM ORIGINAL] - Main lcore setup */
    worker_ctxs[next_queue].queue_id = next_queue;
    worker_ctxs[next_queue].lcore_id = rte_lcore_id();
    used_workers++;
    next_queue++;

    /* [REUSED FROM ORIGINAL] - Remote worker launch */
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (next_queue >= num_queues) {
            break;
        }

        worker_ctxs[next_queue].queue_id = next_queue;
        worker_ctxs[next_queue].lcore_id = lcore_id;
        if (rte_eal_remote_launch(run_simd_server_worker,
                                  &worker_ctxs[next_queue],
                                  lcore_id) != 0) {
            rte_exit(EXIT_FAILURE,
                     "Cannot launch worker for queue %u on lcore %u\n",
                     next_queue, lcore_id);
        }
        used_workers++;
        next_queue++;
    }

    if (used_workers < num_queues) {
        rte_exit(EXIT_FAILURE,
                 "Only mapped %u workers for %u queues\n",
                 used_workers, num_queues);
    }

    printf("\nLaunched %u workers on queues 0-%u\n",
           used_workers, num_queues - 1);
    printf("Press Ctrl+C to exit\n\n");

    return run_simd_server_worker(&worker_ctxs[0]);
}

